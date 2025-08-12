#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

// Test program to detect CUDA compute capability and FP4 support
int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    
    std::cout << "CUDA Device Count: " << device_count << std::endl;
    
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        std::cout << "\n=== Device " << i << " ===" << std::endl;
        std::cout << "Name: " << prop.name << std::endl;
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Memory Clock Rate (MHz): " << prop.memoryClockRate / 1000 << std::endl;
        std::cout << "Memory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
        
        // Check for Blackwell (Compute 10.x) which should support native FP4
        if (prop.major >= 10) {
            std::cout << ">>> Blackwell Architecture Detected - FP4 Tensor Cores Likely Supported" << std::endl;
        } else if (prop.major >= 9) {
            std::cout << ">>> Hopper/Ada Architecture - FP8 Tensor Cores Supported" << std::endl;
        } else if (prop.major >= 8) {
            std::cout << ">>> Ampere Architecture - INT8 Tensor Cores Supported" << std::endl;
        } else if (prop.major >= 7) {
            std::cout << ">>> Turing/Volta Architecture - Mixed Precision Supported" << std::endl;
        } else {
            std::cout << ">>> Older Architecture - Limited Tensor Support" << std::endl;
        }
    }
    
    return 0;
}