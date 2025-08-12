#include <cuda_runtime.h>
#include <iostream>

// Simple test that checks architecture without allocating memory
__global__ void check_arch() {
    #ifdef __CUDA_ARCH__
    printf("CUDA Architecture: %d\n", __CUDA_ARCH__);
    
    if (__CUDA_ARCH__ >= 1200) {
        printf("Blackwell (sm_%d) - FP4 tensor cores should be supported\n", __CUDA_ARCH__);
    } else if (__CUDA_ARCH__ >= 900) {
        printf("Hopper (sm_%d) - FP8 tensor cores supported\n", __CUDA_ARCH__);
    } else if (__CUDA_ARCH__ >= 800) {
        printf("Ampere (sm_%d) - INT8 tensor cores supported\n", __CUDA_ARCH__);
    } else {
        printf("Older architecture (sm_%d) - Limited tensor support\n", __CUDA_ARCH__);
    }
    
    // Check for specific instruction support
    #if __CUDA_ARCH__ >= 1200
    printf("FP4 tensor instructions likely available\n");
    #elif __CUDA_ARCH__ >= 900  
    printf("FP8 tensor instructions available\n");
    #elif __CUDA_ARCH__ >= 800
    printf("INT8 DP4A instructions available\n");
    #endif
    #endif
}

int main() {
    std::cout << "Checking CUDA Architecture Support" << std::endl;
    
    // Launch with minimal resources
    check_arch<<<1, 1>>>();
    
    cudaDeviceSynchronize();
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "Architecture check completed" << std::endl;
    return 0;
}