#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

// Test if experimental FP4 precision is available
__global__ void test_fp4_precision() {
    printf("Testing FP4 precision availability...\n");
    
    // Try to create FP4 fragments - these might not exist yet
    #ifdef __CUDA_ARCH__
    #if __CUDA_ARCH__ >= 1200
    
    // Test if we can use experimental FP4 precision
    // This might not compile if FP4 is not implemented yet
    
    #ifdef CUDA_EXPERIMENTAL_FP4_SUPPORTED
    // Hypothetical FP4 fragment creation
    wmma::fragment<wmma::matrix_a, 16, 16, 16, wmma::experimental::precision::e2m1, wmma::row_major> a_fp4;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, wmma::experimental::precision::e2m1, wmma::col_major> b_fp4;
    printf("FP4 fragments created successfully!\n");
    #else
    printf("FP4 experimental precision not available in CUDA headers\n");
    #endif
    
    // Test FP8 instead (E4M3, E5M2)
    #ifdef CUDA_EXPERIMENTAL_FP8_SUPPORTED  
    wmma::fragment<wmma::matrix_a, 16, 16, 16, wmma::experimental::precision::e4m3, wmma::row_major> a_fp8;
    printf("FP8 E4M3 fragment available\n");
    #else
    printf("FP8 experimental precision not available\n");
    #endif
    
    #endif
    #endif
}

// Test compilation with different precision formats
template<typename T>
__device__ void test_precision_support() {
    // This template will help us test what precisions compile
    T test_value = T(1.0f);
    printf("Precision test value: %f\n", (float)test_value);
}

__global__ void test_available_precisions() {
    printf("Testing available precision formats...\n");
    
    // Test standard precisions
    test_precision_support<half>();
    test_precision_support<float>();
    
    // Test if __nv_fp8_e4m3 is available
    #ifdef __CUDA_EXPERIMENTAL_FP8__
    test_precision_support<__nv_fp8_e4m3>();
    printf("FP8 E4M3 supported\n");
    #endif
    
    #ifdef __CUDA_EXPERIMENTAL_FP8_E5M2__  
    test_precision_support<__nv_fp8_e5m2>();
    printf("FP8 E5M2 supported\n");
    #endif
}

int main() {
    std::cout << "Testing FP4/FP8 Precision Support in CUDA 12.9" << std::endl;
    
    test_fp4_precision<<<1, 1>>>();
    test_available_precisions<<<1, 1>>>();
    
    cudaDeviceSynchronize();
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
    }
    
    std::cout << "Precision test completed" << std::endl;
    return 0;
}