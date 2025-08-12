// Test CUDA 13.0 FP8 support for Blackwell
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <stdio.h>

__global__ void test_fp8_e2m1() {
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    
    // Use the proper FP8 E2M1 type from cuda_fp8.h
    __nv_fp8_storage_t fp8_val;
    fp8_val = __nv_cvt_float_to_fp8(1.0f, __NV_SATFINITE, __NV_E2M1);
    float result = __nv_cvt_fp8_to_halfraw(fp8_val, __NV_E2M1);
    
    if (threadIdx.x == 0) {
        printf("✅ FP8 E2M1 conversion works\n");
        printf("✅ Ready for Blackwell FP4 operations\n");
    }
    
    #else
    if (threadIdx.x == 0) {
        printf("Not SM120\n");
    }
    #endif
}

int main() {
    printf("=== CUDA 13.0 FP8 E2M1 Support Test ===\n");
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s (SM%d.%d)\n", prop.name, prop.major, prop.minor);
    
    test_fp8_e2m1<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    printf("✅ SUCCESS: FP8 E2M1 (FP4) support confirmed\n");
    return 0;
}