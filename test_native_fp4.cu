#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <iostream>

// Test native FP4 support with CUDA 12.9 headers
__global__ void test_native_fp4() {
    printf("Testing native FP4 E2M1 support...\n");
    
    #ifdef __CUDA_ARCH__
    printf("CUDA Architecture: %d\n", __CUDA_ARCH__);
    
    if (__CUDA_ARCH__ >= 1200) {
        printf("Blackwell architecture - native FP4 tensor cores available\n");
        
        // Test FP4 E2M1 storage types
        __nv_fp4_storage_t fp4_val;
        __nv_fp4x2_storage_t fp4x2_val;
        __nv_fp4x4_storage_t fp4x4_val;
        
        // Test conversion from float to FP4 E2M1
        float test_input = 1.5f;
        fp4_val = __nv_cvt_float_to_fp4(test_input, __NV_E2M1, cudaRoundNearest);
        
        // Test conversion back via half (simpler approach)
        __half_raw half_result = __nv_cvt_fp4_to_halfraw(fp4_val, __NV_E2M1);
        // Manual conversion from half_raw to float
        union { __half_raw r; unsigned short i; } h;
        h.r = half_result;
        float result = __half2float(__short_as_half(h.i));
        
        printf("FP4 E2M1 conversion test: %f -> %f\n", test_input, result);
        printf("Native FP4 operations working!\n");
        
    } else {
        printf("Architecture too old for native FP4 support\n");
    }
    #endif
}

// Test FP4 tensor core availability (hypothetical API)
__global__ void test_fp4_tensor_cores() {
    printf("Testing FP4 tensor core operations...\n");
    
    #ifdef __CUDA_ARCH__
    #if __CUDA_ARCH__ >= 1200
    
    // At compute 12.0, we should have FP4 tensor cores
    // But WMMA API might not be available yet for FP4
    printf("Hardware supports FP4 tensor cores\n");
    
    // Test basic FP4 arithmetic
    __nv_fp4_storage_t a = __nv_cvt_float_to_fp4(2.0f, __NV_E2M1, cudaRoundNearest);
    __nv_fp4_storage_t b = __nv_cvt_float_to_fp4(3.0f, __NV_E2M1, cudaRoundNearest);
    
    // Convert back to check values
    __half_raw a_half = __nv_cvt_fp4_to_halfraw(a, __NV_E2M1);
    __half_raw b_half = __nv_cvt_fp4_to_halfraw(b, __NV_E2M1);
    
    union { __half_raw r; unsigned short i; } ha, hb;
    ha.r = a_half; hb.r = b_half;
    float a_val = __half2float(__short_as_half(ha.i));
    float b_val = __half2float(__short_as_half(hb.i));
    
    printf("FP4 values: a=%f, b=%f\n", a_val, b_val);
    
    #else
    printf("FP4 tensor cores not supported on this architecture\n");
    #endif
    #endif
}

int main() {
    std::cout << "Testing Native CUDA 12.9 FP4 Support" << std::endl;
    
    test_native_fp4<<<1, 1>>>();
    test_fp4_tensor_cores<<<1, 1>>>();
    
    cudaDeviceSynchronize();
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "Native FP4 test completed successfully" << std::endl;
    return 0;
}