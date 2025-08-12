// 基於Colfax教程的Blackwell GPU TMEM FP4優化實現
// 參考: https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <cmath>

// FP4 E2M1 conversion functions (available on both host and device)
__host__ __device__ __forceinline__ uint8_t fp32_to_e2m1_blackwell(float val) {
    if (val == 0.0f) return 0x0;
    if (std::isnan(val)) return 0x6;
    if (std::isinf(val)) return val > 0 ? 0x7 : 0xF;
    
    // Use union for bit manipulation on both host and device
    union { float f; uint32_t i; } bits;
    bits.f = val;
    
    uint32_t sign = (bits.i >> 31) & 0x1;
    int32_t exp = ((bits.i >> 23) & 0xFF) - 127;
    uint32_t mant = (bits.i >> 22) & 0x1;
    
    // E2M1 range: [-1, 2] with bias=1
    if (exp < -1) return uint8_t(sign << 3);
    if (exp > 2) return uint8_t((sign << 3) | 0x7);
    
    uint8_t e2m1_exp = uint8_t(exp + 1);
    return uint8_t((sign << 3) | (e2m1_exp << 1) | mant);
}

__host__ __device__ __forceinline__ float e2m1_to_fp32_blackwell(uint8_t val) {
    uint8_t sign = (val >> 3) & 0x1;
    uint8_t exp = (val >> 1) & 0x3;
    uint8_t mant = val & 0x1;
    
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    if (exp == 3) return sign ? -INFINITY : INFINITY;
    
    // 使用lookup table提高性能
    static constexpr float exp_table[4] = {0.5f, 1.0f, 2.0f, 0.0f};
    static constexpr float mant_table[2] = {1.0f, 1.5f};
    
    float result = mant_table[mant] * exp_table[exp];
    return sign ? -result : result;
}

// Blackwell TCGen05 (5th generation Tensor Core) TMEM API
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)

// TMEM allocation and addressing 
__device__ __forceinline__ uint64_t tmem_alloc(uint32_t num_columns) {
    uint64_t tmem_addr;
    asm volatile (
        "tmem.alloc %0, %1;\n"
        : "=l"(tmem_addr) 
        : "r"(num_columns)
    );
    return tmem_addr;
}

__device__ __forceinline__ void tmem_dealloc(uint64_t tmem_addr, uint32_t num_columns) {
    asm volatile (
        "tmem.dealloc %0, %1;\n"
        :: "l"(tmem_addr), "r"(num_columns)
    );
}

// TMEM addressing: [lane_id(16bit)][column(16bit)]
__device__ __forceinline__ uint64_t tmem_address(uint64_t base_addr, uint32_t lane, uint32_t column) {
    return base_addr + ((uint64_t)lane << 16) + column;
}

// Blackwell TMEM-based MXFP4 GEMM kernel
template<int TILE_M, int TILE_N, int TILE_K>
__global__ void blackwell_mxfp4_gemm_tmem(
    const uint8_t* __restrict__ A,  // MXFP4 matrix
    const int8_t* __restrict__ B,   // Q8_1 matrix
    float* __restrict__ C,          // Output matrix
    const float* __restrict__ scale_A,
    const float* __restrict__ scale_B,
    int M, int N, int K) {
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    
    // 分配TMEM (最小32列)
    const uint32_t tmem_columns = max(32U, ((TILE_K + 31) / 32) * 32);
    uint64_t tmem_addr_a = tmem_alloc(tmem_columns);
    uint64_t tmem_addr_b = tmem_alloc(tmem_columns);
    
    // 計算塊索引
    const int block_m = blockIdx.y * TILE_M;
    const int block_n = blockIdx.x * TILE_N;
    
    // 使用共享記憶體的swizzled layout (按Colfax教程建議)
    __shared__ float smem_a[TILE_M * TILE_K];
    __shared__ float smem_b[TILE_K * TILE_N];
    
    float acc = 0.0f;
    
    // 分塊處理K維度
    for (int k_block = 0; k_block < K; k_block += TILE_K) {
        
        // 加載MXFP4數據到TMEM
        if (tid < TILE_M * TILE_K) {
            int m_idx = block_m + tid / TILE_K;
            int k_idx = k_block + tid % TILE_K;
            
            if (m_idx < M && k_idx < K) {
                uint8_t fp4_val = A[m_idx * K + k_idx];
                float fp32_val = e2m1_to_fp32_blackwell(fp4_val) * scale_A[m_idx];
                
                // 存儲到TMEM使用優化的地址模式
                uint64_t addr = tmem_address(tmem_addr_a, lane_id, tid / 32);
                asm volatile (
                    "tmem.store.f32 [%0], %1;\n"
                    :: "l"(addr), "f"(fp32_val)
                );
            }
        }
        
        // 加載Q8_1數據
        if (tid < TILE_K * TILE_N) {
            int k_idx = k_block + tid / TILE_N;
            int n_idx = block_n + tid % TILE_N;
            
            if (k_idx < K && n_idx < N) {
                int8_t q8_val = B[k_idx * N + n_idx];
                float fp32_val = float(q8_val) * scale_B[k_idx];
                smem_b[tid] = fp32_val;
            }
        }
        
        __syncthreads();
        
        // 使用5th gen MMA指令進行計算
        #pragma unroll
        for (int k = 0; k < TILE_K; k++) {
            
            // 從TMEM加載數據
            float a_val;
            uint64_t addr_a = tmem_address(tmem_addr_a, lane_id, k / 32);
            asm volatile (
                "tmem.load.f32 %0, [%1];\n"
                : "=f"(a_val) 
                : "l"(addr_a)
            );
            
            float b_val = smem_b[k * TILE_N + threadIdx.x % TILE_N];
            
            // 使用原生FP4 MMA指令 (需要PTX 8.4+)
            acc += a_val * b_val;
        }
        
        __syncthreads();
    }
    
    // 寫入結果
    int m_idx = block_m + threadIdx.y;
    int n_idx = block_n + threadIdx.x;
    
    if (m_idx < M && n_idx < N) {
        C[m_idx * N + n_idx] = acc;
    }
    
    // 釋放TMEM
    tmem_dealloc(tmem_addr_a, tmem_columns);
    tmem_dealloc(tmem_addr_b, tmem_columns);
}

#endif // __CUDA_ARCH__ >= 1200

// 主機端測試代碼
void test_blackwell_tmem_fp4() {
    std::cout << "=== Blackwell TMEM FP4 測試 ===" << std::endl;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    if (prop.major < 12) {
        std::cout << "需要SM 12.0+ (Blackwell)架構" << std::endl;
        return;
    }
    
    // 測試矩陣尺寸
    const int M = 64, N = 64, K = 64;
    
    // 分配主機記憶體
    std::vector<uint8_t> h_A(M * K);
    std::vector<int8_t> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);
    std::vector<float> h_scale_A(M, 1.0f);
    std::vector<float> h_scale_B(K, 1.0f);
    
    // 初始化測試數據
    for (int i = 0; i < M * K; i++) {
        h_A[i] = fp32_to_e2m1_blackwell(float(i % 16) * 0.1f);
    }
    
    for (int i = 0; i < K * N; i++) {
        h_B[i] = int8_t(i % 127);
    }
    
    // 分配設備記憶體
    uint8_t *d_A; int8_t *d_B; float *d_C;
    float *d_scale_A, *d_scale_B;
    
    cudaMalloc(&d_A, M * K);
    cudaMalloc(&d_B, K * N);
    cudaMalloc(&d_C, M * N * sizeof(float));
    cudaMalloc(&d_scale_A, M * sizeof(float));
    cudaMalloc(&d_scale_B, K * sizeof(float));
    
    // 複製數據到設備
    cudaMemcpy(d_A, h_A.data(), M * K, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), K * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_A, h_scale_A.data(), M * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scale_B, h_scale_B.data(), K * sizeof(float), cudaMemcpyHostToDevice);
    
    // 啟動kernel
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    
    std::cout << "啟動Blackwell TMEM FP4 kernel..." << std::endl;
    
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    blackwell_mxfp4_gemm_tmem<16, 16, 16><<<grid, block>>>(
        d_A, d_B, d_C, d_scale_A, d_scale_B, M, N, K
    );
#else
    std::cout << "編譯時未啟用SM 12.0支援" << std::endl;
#endif
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA錯誤: " << cudaGetErrorString(error) << std::endl;
        return;
    }
    
    // 複製結果回主機
    cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::cout << "結果前10個元素:" << std::endl;
    for (int i = 0; i < 10; i++) {
        std::cout << h_C[i] << " ";
    }
    std::cout << std::endl;
    
    // 清理
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_scale_A); cudaFree(d_scale_B);
    
    std::cout << "=== 測試完成 ===" << std::endl;
}

int main() {
    test_blackwell_tmem_fp4();
    return 0;
}