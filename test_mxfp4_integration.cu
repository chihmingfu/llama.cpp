// MXFP4 Tensor Core 測試程序 - RTX 5070 Blackwell 專用
// 測試 CUDA 13.0 原生 FP4 指令和 Tensor Core 支持

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <chrono>

// 檢查是否可以使用 cuda_fp4.h
#ifdef __CUDACC_VER_MAJOR__ 
#if __CUDACC_VER_MAJOR__ >= 13
    // 嘗試包含 CUDA 13.0 FP4 頭文件
    #if __has_include(<cuda_fp4.h>)
        #include <cuda_fp4.h>
        #define CUDA_FP4_AVAILABLE 1
    #else
        #define CUDA_FP4_AVAILABLE 0
    #endif
#else
    #define CUDA_FP4_AVAILABLE 0
#endif
#else
    #define CUDA_FP4_AVAILABLE 0
#endif

// MXFP4 E2M1 格式定義
struct __align__(1) mxfp4_t {
    uint8_t data : 4;
    
    __host__ __device__ __forceinline__ mxfp4_t() : data(0) {}
    __host__ __device__ __forceinline__ mxfp4_t(uint8_t val) : data(val & 0xF) {}
    
    // E2M1 到 FP32 轉換
    __device__ __forceinline__ float to_float() const {
        uint8_t sign = (data >> 3) & 0x1;
        uint8_t exp = (data >> 1) & 0x3;  
        uint8_t mant = data & 0x1;
        
        if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
        if (exp == 3) {
            // NaN 或 Inf
            return sign ? (mant ? -__int_as_float(0x7fc00000) : -__int_as_float(0x7f800000)) : 
                          (mant ? __int_as_float(0x7fc00000) : __int_as_float(0x7f800000));
        }
        
        // 正常值計算 - 使用預計算查找表
        const float exp_vals[4] = {0.5f, 1.0f, 2.0f, 0.0f};
        const float mant_vals[2] = {1.0f, 1.5f};
        
        float result = mant_vals[mant] * exp_vals[exp];
        return sign ? -result : result;
    }
};

// CUDA 設備信息檢查
__global__ void check_device_capability() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
#ifdef __CUDA_ARCH__
        int sm_major = __CUDA_ARCH__ / 100;
        int sm_minor = (__CUDA_ARCH__ % 100) / 10;
        printf("Device SM: %d.%d\n", sm_major, sm_minor);
        printf("Blackwell Support (SM 12.0+): %s\n", 
               (__CUDA_ARCH__ >= 1200) ? "YES" : "NO");
#else
        printf("Device SM: Unknown (host code)\n");
#endif
        
#if CUDA_FP4_AVAILABLE
        printf("CUDA FP4 Header Available: YES\n");
#else
        printf("CUDA FP4 Header Available: NO\n");
#endif
    }
}

// 基礎 MXFP4 向量點積測試
__global__ void test_mxfp4_dot_product(
    const mxfp4_t* a, 
    const float* b, 
    float* result,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // 簡單的 FP4 到 FP32 轉換和乘法
    float fp4_val = a[idx].to_float();
    result[idx] = fp4_val * b[idx];
}

// Tensor Core 模擬測試 (為未來的真實實現做準備)
__global__ void test_mxfp4_tensor_core_simulation(
    const mxfp4_t* a_fp4,
    const half* b_fp16, 
    float* c_fp32,
    int M, int N, int K
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    
    // 模擬 Tensor Core 運算 - 將來會替換為真實的 TCGen05 指令
    for (int k = 0; k < K; k++) {
        float a_val = a_fp4[row * K + k].to_float();
        float b_val = __half2float(b_fp16[k * N + col]);
        sum += a_val * b_val;
    }
    
    c_fp32[row * N + col] = sum;
}

// 性能測試函數
void benchmark_mxfp4_performance() {
    const int M = 1024, N = 1024, K = 1024;
    
    // 分配設備內存
    mxfp4_t* d_a;
    half* d_b;
    float* d_c;
    
    cudaMalloc(&d_a, M * K * sizeof(mxfp4_t));
    cudaMalloc(&d_b, K * N * sizeof(half));
    cudaMalloc(&d_c, M * N * sizeof(float));
    
    // 初始化數據
    std::vector<mxfp4_t> h_a(M * K);
    std::vector<half> h_b(K * N);
    
    for (int i = 0; i < M * K; i++) {
        h_a[i] = mxfp4_t(i % 16); // 簡單的測試模式
    }
    
    for (int i = 0; i < K * N; i++) {
        h_b[i] = __float2half(0.5f); // 固定值測試
    }
    
    cudaMemcpy(d_a, h_a.data(), M * K * sizeof(mxfp4_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), K * N * sizeof(half), cudaMemcpyHostToDevice);
    
    // 設置網格和塊維度
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    
    // 預熱
    test_mxfp4_tensor_core_simulation<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    cudaDeviceSynchronize();
    
    // 性能測試
    auto start = std::chrono::high_resolution_clock::now();
    
    const int num_iterations = 100;
    for (int i = 0; i < num_iterations; i++) {
        test_mxfp4_tensor_core_simulation<<<grid, block>>>(d_a, d_b, d_c, M, N, K);
    }
    
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double ops = (double)M * N * K * 2.0 * num_iterations; // 乘加運算
    double gflops = ops / (duration.count() / 1000.0) / 1e9;
    
    std::cout << "MXFP4 Performance Test Results:" << std::endl;
    std::cout << "Matrix Size: " << M << "x" << N << "x" << K << std::endl;
    std::cout << "Time: " << duration.count() / 1000.0 << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;
    
    // 清理
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

int main() {
    std::cout << "=== MXFP4 Tensor Core 測試程序 ===" << std::endl;
    
    // 檢查 CUDA 設備
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (device_count == 0) {
        std::cerr << "未找到 CUDA 設備！" << std::endl;
        return -1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    std::cout << "設備: " << prop.name << std::endl;
    std::cout << "計算能力: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "CUDA 編譯器版本: " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << std::endl;
    
    if (prop.major < 12) {
        std::cout << "警告: 此設備不支持 Blackwell Tensor Core (需要 SM 12.0+)" << std::endl;
    }
    
    // 執行設備端檢查
    std::cout << "\n--- 設備端檢查 ---" << std::endl;
    check_device_capability<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    // 執行性能測試
    std::cout << "\n--- 性能測試 ---" << std::endl;
    benchmark_mxfp4_performance();
    
    // 檢查 CUDA 錯誤
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA 錯誤: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }
    
    std::cout << "\n測試完成！" << std::endl;
    return 0;
}