#include <cuda_runtime.h>
#include <mma.h>
#include <iostream>

using namespace nvcuda;

// Test FP4 WMMA API availability
__global__ void test_wmma_fp4() {
    // Try to test FP4 tensor core availability
    printf("GPU: Testing WMMA FP4 support...\n");
    
    // Check if experimental FP4 precision is available
    #ifdef __CUDA_ARCH__
    printf("CUDA Architecture: %d\n", __CUDA_ARCH__);
    
    if (__CUDA_ARCH__ >= 1200) {  // Blackwell (compute 12.0)
        printf("Blackwell architecture detected - FP4 should be supported\n");
    } else if (__CUDA_ARCH__ >= 900) {  // Hopper
        printf("Hopper architecture - FP8 supported\n");
    } else {
        printf("Older architecture - Limited precision support\n");
    }
    #endif
}

// Test basic tensor core operations
__global__ void test_basic_wmma() {
    printf("Testing basic WMMA operations...\n");
    
    // Test if we can use half precision WMMA (baseline)
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    
    // Initialize accumulator
    wmma::fill_fragment(c_frag, 0.0f);
    
    printf("Basic WMMA FP16 fragments created successfully\n");
}

int main() {
    std::cout << "Testing CUDA WMMA FP4 Support" << std::endl;
    
    // Launch kernel to test on GPU
    test_wmma_fp4<<<1, 32>>>();
    test_basic_wmma<<<1, 32>>>();
    
    cudaDeviceSynchronize();
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cout << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "WMMA test completed successfully" << std::endl;
    return 0;
}