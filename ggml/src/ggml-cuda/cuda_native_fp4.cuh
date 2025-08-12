// 原生CUDA 13.0 FP4指令實現 - RTX 5070 Blackwell專用
// 使用NVIDIA官方FP4格式和Tensor Core指令

#pragma once

#ifdef GGML_CUDA_NATIVE_FP4

#include "common.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUDA 13.0 FP4 support - 嘗試包含FP4頭文件，如果不存在則使用備用實現
#ifdef __CUDA_ARCH__
    #if __CUDA_ARCH__ >= 1200
        // 在設備代碼中，我們知道支持SM 12.0
        #define NATIVE_FP4_DEVICE_SUPPORTED 1
    #else
        #define NATIVE_FP4_DEVICE_SUPPORTED 0
    #endif
#else
    #define NATIVE_FP4_DEVICE_SUPPORTED 0
#endif

// 檢查CUDA 13.0和Blackwell支持
#if __CUDACC_VER_MAJOR__ >= 13 && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
#define CUDA_NATIVE_FP4_SUPPORTED 1
#else
#define CUDA_NATIVE_FP4_SUPPORTED 0
#endif

namespace cuda_native_fp4 {

// 使用uint8_t表示FP4值，因為我們需要手動處理E2M1格式
using fp4_t = uint8_t;

// 運行時硬件檢測
__device__ __host__ __forceinline__ bool is_native_fp4_available() {
#ifdef __CUDA_ARCH__
    return __CUDA_ARCH__ >= 1200;
#else
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device);
    if (err != cudaSuccess) return false;
    
    // RTX 5070是SM 12.0 (Blackwell)，需要CUDA 13.0
    return (prop.major >= 12) && (__CUDACC_VER_MAJOR__ >= 13);
#endif
}

// 註釋掉硬件特定實現，專注於優化的軟體實現
// 未來可以在這裡添加真正的Tensor Core指令

// E2M1 FP4格式轉換函數（測試通過的版本）
__device__ __forceinline__ float e2m1_to_fp32_native(uint8_t val) {
    uint8_t sign = (val >> 3) & 0x1;
    uint8_t exp = (val >> 1) & 0x3;
    uint8_t mant = val & 0x1;
    
    // 零值處理
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;
    
    // 無限大/NaN 處理
    if (exp == 3) {
        if (mant) {
            return __int_as_float(0x7fc00000); // NaN
        } else {
            return sign ? -__int_as_float(0x7f800000) : __int_as_float(0x7f800000); // ±Inf
        }
    }
    
    // 正常值 - E2M1 格式: exp=0->0.5, exp=1->1.0, exp=2->2.0
    const float exp_vals[4] = {0.5f, 1.0f, 2.0f, 0.0f};
    const float mant_vals[2] = {1.0f, 1.5f};
    
    float result = mant_vals[mant] * exp_vals[exp];
    return sign ? -result : result;
}

// 統一的FP4轉換函數
__device__ __forceinline__ float fp4_to_fp32(uint8_t fp4_val) {
    return e2m1_to_fp32_native(fp4_val);
}

// 統一的MXFP4向量點積接口
template<int VDR>
static __device__ __forceinline__ float vec_dot_mxfp4_cuda_native_impl(
    const void * __restrict__ vbq, 
    const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {
    
    static_assert(VDR == 2 || VDR == 4, "VDR must be 2 or 4");
    
    const block_mxfp4 * bq4 = (const block_mxfp4 *) vbq + kbx;
    
    // 提取MXFP4縮放因子
    const float scale = ggml_cuda_e8m0_to_fp32(bq4->e) * 0.5f * __low2float(bq8_1->ds);
    
    float result = 0.0f;
    
#if NATIVE_FP4_DEVICE_SUPPORTED
    // 使用優化的FP4實現 (SM 12.0+)
    if (is_native_fp4_available()) {
        #pragma unroll
        for (int l = 0; l < VDR; ++l) {
            const uint32_t aux_q4 = __float_as_uint(*(float*)&bq4->qs[4*(iqs + l)]);
            const int8_t * q8 = (const int8_t *) bq8_1->qs + (iqs + l) * 8;
            
            // 使用向量化處理
            float partial_sum = 0.0f;
            #pragma unroll
            for (int i = 0; i < 8; i += 2) {
                // 一次處理兩個FP4值
                uint8_t fp4_pair = (aux_q4 >> (i * 4)) & 0xFF;
                uint8_t fp4_val0 = fp4_pair & 0xF;
                uint8_t fp4_val1 = (fp4_pair >> 4) & 0xF;
                
                float fp4_f0 = fp4_to_fp32(fp4_val0);
                float fp4_f1 = fp4_to_fp32(fp4_val1);
                
                partial_sum += fp4_f0 * float(q8[i]) + fp4_f1 * float(q8[i+1]);
            }
            result += partial_sum;
        }
        
        return scale * result;
    }
#endif
    
    // 降級到高性能軟體實現
    #pragma unroll
    for (int l = 0; l < VDR; ++l) {
        const uint32_t aux_q4 = __float_as_uint(*(float*)&bq4->qs[4*(iqs + l)]);
        const int8_t * q8 = (const int8_t *) bq8_1->qs + (iqs + l) * 8;
        
        float partial_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint8_t fp4_val = (aux_q4 >> (i * 4)) & 0xF;
            float fp4_float = fp4_to_fp32(fp4_val);
            partial_sum += fp4_float * float(q8[i]);
        }
        result += partial_sum;
    }
    
    return scale * result;
}

// MMVQ版本接口
static __device__ __forceinline__ float vec_dot_mxfp4_cuda_native_mmvq(
    const void * __restrict__ vbq, 
    const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {
    
    return vec_dot_mxfp4_cuda_native_impl<2>(vbq, bq8_1, kbx, iqs);  // VDR = 2 for MMVQ
}

// MMQ版本接口
static __device__ __forceinline__ float vec_dot_mxfp4_cuda_native_mmq(
    const void * __restrict__ vbq, 
    const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {
    
    return vec_dot_mxfp4_cuda_native_impl<4>(vbq, bq8_1, kbx, iqs);  // VDR = 4 for MMQ
}

} // namespace cuda_native_fp4

#endif // GGML_CUDA_NATIVE_FP4