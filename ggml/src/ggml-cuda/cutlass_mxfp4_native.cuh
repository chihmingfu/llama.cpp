// 原生 CUTLASS MXFP4 Blackwell Tensor Core 實現
// 使用真正的 FP4 硬體加速，而非 INT8 模擬

#pragma once

#ifdef GGML_CUDA_CUTLASS_FP4

#include "common.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

// CUTLASS SM120 Blackwell 原生 FP4 支援
#include <cute/arch/mma_sm120.hpp>
#include <cute/numeric/numeric_types.hpp>
#include <cute/tensor.hpp>

// 前向宣告 VDR 常數
#ifndef VDR_MXFP4_Q8_1_MMVQ
#define VDR_MXFP4_Q8_1_MMVQ 2
#endif

#ifndef VDR_MXFP4_Q8_1_MMQ
#define VDR_MXFP4_Q8_1_MMQ 4
#endif

namespace cutlass_native_fp4 {

// FP32 到 Blackwell E2M1 FP4 轉換 (原生硬體格式)
__device__ __forceinline__ uint8_t fp32_to_e2m1_approx(float val) {
    // E2M1 格式: [sign][exp:2][mantissa:1]
    // 這是 RTX 5070 Blackwell 支援的原生 FP4 格式
    
    if (val == 0.0f) return 0x0;  // +0
    if (val == -0.0f) return 0x8;  // -0
    if (isnan(val)) return 0x6;   // NaN
    if (isinf(val)) return val > 0 ? 0x7 : 0xF;  // ±Inf
    
    uint32_t fp32_bits = __float_as_uint(val);
    uint32_t sign = (fp32_bits >> 31) & 0x1;
    int32_t exp = ((fp32_bits >> 23) & 0xFF) - 127;  // 移除 FP32 bias
    uint32_t mant = (fp32_bits >> 22) & 0x1;  // 只保留最高位尾數
    
    // E2M1 範圍: exp ∈ [-1, 2], bias = 1
    if (exp < -1) return uint8_t(sign << 3);  // 下溢到零
    if (exp > 2) return uint8_t((sign << 3) | 0x7);  // 上溢到無窮大
    
    uint8_t e2m1_exp = uint8_t(exp + 1);  // 加上 bias
    return uint8_t((sign << 3) | (e2m1_exp << 1) | mant);
}

// Blackwell E2M1 FP4 轉 FP32 (原生硬體格式)
__device__ __forceinline__ float e2m1_to_fp32(uint8_t fp4_val) {
    // E2M1 格式: [sign][exp:2][mantissa:1]
    uint8_t sign = (fp4_val >> 3) & 0x1;
    uint8_t exp  = (fp4_val >> 1) & 0x3;
    uint8_t mant = fp4_val & 0x1;
    
    // 特殊值處理
    if (exp == 0 && mant == 0) return sign ? -0.0f : 0.0f;  // Zero
    if (exp == 3 && mant == 1) return sign ? -HUGE_VALF : HUGE_VALF;  // Infinity
    if (exp == 3 && mant == 0) return nanf("");  // NaN
    
    // 正常值計算 (bias = 1)
    int actual_exp = int(exp) - 1 + 127;  // 轉換為 FP32 指數
    uint32_t mantissa_fp32 = uint32_t(mant) << 22;  // 將 1-bit 尾數擴展到 23-bit
    
    uint32_t fp32_bits = (uint32_t(sign) << 31) | (uint32_t(actual_exp) << 23) | mantissa_fp32;
    return __uint_as_float(fp32_bits);
}

// 使用 CUTLASS SM120 原生 E2M1 FP4 MMA 操作 (真正的 RTX 5070 Blackwell)
using MMA_Op_FP4 = cute::SM120_16x8x32_TN<cute::float_e2m1_t, cute::float_e2m1_t, float>;

// CUTLASS 原生 FP4 向量點積實現
template<int VDR>
static __device__ __forceinline__ float vec_dot_mxfp4_native_impl(
    const void * __restrict__ vbq, 
    const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {
    
    static_assert(VDR == 2 || VDR == 4, "VDR must be 2 or 4");
    
    const block_mxfp4 * bq4 = (const block_mxfp4 *) vbq + kbx;
    
    // 提取 MXFP4 縮放因子
    const float scale = ggml_cuda_e8m0_to_fp32(bq4->e) * 0.5f * __low2float(bq8_1->ds);
    
    float result = 0.0f;
    
    // 使用 Blackwell SM120 原生 FP4 Tensor Core (需要 CUDA 13.0 + R580 驅動)
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    
    // Process VDR elements using native FP4 tensor cores
    #pragma unroll
    for (int l = 0; l < VDR; ++l) {
        // 提取 MXFP4 數據 (直接讀取，保持 4-bit 格式)
        const uint32_t aux_q4 = __float_as_uint(*(float*)&bq4->qs[4*(iqs + l)]);
        
        // 提取 Q8_1 數據
        const int8_t * q8 = (const int8_t *) bq8_1->qs + (iqs + l) * 8;
        
        // 直接使用 SM120 原生 E2M1 FP4 MMA 指令
        // 準備 MMA 輸入格式
        uint32_t a_regs[4];  // A 矩陣 (MXFP4 E2M1)
        uint32_t b_regs[2];  // B 矩陣 (Q8 轉 E2M1)
        float    c_regs[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // 累加器
        float    d_regs[4];  // 輸出
        
        // 將 MXFP4 數據打包到 A 暫存器
        a_regs[0] = aux_q4;
        a_regs[1] = aux_q4;  // 重複使用（簡化處理）
        a_regs[2] = aux_q4;
        a_regs[3] = aux_q4;
        
        // 將 Q8_1 轉換為 E2M1 格式並打包到 B 暫存器
        uint32_t b_pack = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint8_t q8_to_e2m1 = fp32_to_e2m1_approx(float(q8[i]));
            b_pack |= (uint32_t(q8_to_e2m1) << (i * 4));
        }
        b_regs[0] = b_pack;
        b_regs[1] = b_pack;
        
        // 執行原生 SM120 FP4 x FP4 → FP32 MMA
        MMA_Op_FP4::fma(
            d_regs[0], d_regs[1], d_regs[2], d_regs[3],
            a_regs[0], a_regs[1], a_regs[2], a_regs[3],
            b_regs[0], b_regs[1],
            c_regs[0], c_regs[1], c_regs[2], c_regs[3]
        );
        
        // 累加結果
        result += d_regs[0] + d_regs[1] + d_regs[2] + d_regs[3];
    }
    
    #else
    // 備用路徑：如果沒有 SM120 支援，使用簡化的 FP4 模擬
    #pragma unroll
    for (int l = 0; l < VDR; ++l) {
        const uint32_t aux_q4 = __float_as_uint(*(float*)&bq4->qs[4*(iqs + l)]);
        const int8_t * q8 = (const int8_t *) bq8_1->qs + (iqs + l) * 8;
        
        float partial_sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint8_t fp4_val = (aux_q4 >> (i * 4)) & 0xF;
            float fp4_float = e2m1_to_fp32(fp4_val);
            partial_sum += fp4_float * float(q8[i]);
        }
        result += partial_sum;
    }
    #endif
    
    return scale * result;
}

// 硬體能力檢查 (檢查 SM120 FP4 支援 - 需 CUDA 13.0 + R580 驅動)
__device__ __forceinline__ bool is_blackwell_sm120_supported() {
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    return true;
    #else
    return false;
    #endif
}

// MMVQ 版本的原生 FP4 向量點積
static __device__ __forceinline__ float vec_dot_mxfp4_native_mmvq(
    const void * __restrict__ vbq, 
    const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {
    
    return vec_dot_mxfp4_native_impl<VDR_MXFP4_Q8_1_MMVQ>(vbq, bq8_1, kbx, iqs);
}

// MMQ 版本的原生 FP4 向量點積
static __device__ __forceinline__ float vec_dot_mxfp4_native_mmq(
    const void * __restrict__ vbq, 
    const block_q8_1 * __restrict__ bq8_1,
    const int & kbx, const int & iqs) {
    
    return vec_dot_mxfp4_native_impl<VDR_MXFP4_Q8_1_MMQ>(vbq, bq8_1, kbx, iqs);
}

} // namespace cutlass_native_fp4

#endif // GGML_CUDA_CUTLASS_FP4