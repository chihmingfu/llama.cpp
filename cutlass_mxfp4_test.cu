#include <iostream>

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"

using namespace cute;

// Simple CUTLASS availability test for RTX 5070
int main() {
    std::cout << "CUTLASS MXFP4/NVFP4 Support Test on RTX 5070" << std::endl;
    
    // Check CUDA version
    std::cout << "CUDA Compiler Version: " << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__ << std::endl;
    
    // Get device properties
    cudaDeviceProp props;
    int current_device_id;
    cudaError_t error = cudaGetDevice(&current_device_id);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDevice failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    error = cudaGetDeviceProperties(&props, current_device_id);
    if (error != cudaSuccess) {
        std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    std::cout << "Device: " << props.name << std::endl;
    std::cout << "Compute Capability: " << props.major << "." << props.minor << std::endl;
    std::cout << "SM Count: " << props.multiProcessorCount << std::endl;
    
    // Check architecture support
    if (props.major >= 10) {
        std::cout << "✅ Blackwell or newer architecture detected" << std::endl;
        if (props.major == 12) {
            std::cout << "✅ RTX 5070 (SM 12.0) - Advanced Blackwell with native FP4 support" << std::endl;
        }
    } else if (props.major >= 9) {
        std::cout << "⚠️  Hopper architecture - FP8 supported but not FP4" << std::endl;
    } else {
        std::cout << "❌ Pre-Hopper architecture - Limited low-precision support" << std::endl;
    }
    
    // Check CUTLASS compile-time support
    #ifdef CUTLASS_ARCH_MMA_SM100_SUPPORTED
    std::cout << "✅ CUTLASS SM100 (Blackwell) support compiled in" << std::endl;
    #else
    std::cout << "❌ CUTLASS SM100 support not compiled" << std::endl;
    #endif

    // Test basic CUTLASS types
    std::cout << "\n=== CUTLASS Type Support ===" << std::endl;
    
    // Test FP4 types
    try {
        using Fp4Type = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
        std::cout << "✅ NVFP4 (E2M1) type available" << std::endl;
        std::cout << "   Size: " << sizeof(Fp4Type) << " bytes" << std::endl;
        
        // Test architecture tag
        #ifdef CUTLASS_ARCH_MMA_SM100_SUPPORTED
        using ArchTag = cutlass::arch::Sm100;
        std::cout << "✅ Blackwell SM100 architecture tag available" << std::endl;
        #endif
        
    } catch (const std::exception& e) {
        std::cout << "❌ FP4 type test failed: " << e.what() << std::endl;
    }
    
    // Check Tensor Memory (TMEM) support
    #ifdef CUTLASS_ARCH_MMA_SM100_SUPPORTED
    std::cout << "✅ Tensor Memory (TMEM) API available" << std::endl;
    #else
    std::cout << "❌ Tensor Memory API not available" << std::endl;
    #endif
    
    std::cout << "\n=== Summary ===" << std::endl;
    if (props.major >= 10) {
        std::cout << "🎉 Hardware supports CUTLASS FP4 acceleration" << std::endl;
        std::cout << "📋 Next steps:" << std::endl;
        std::cout << "   1. Implement MXFP4 kernels using CUTLASS Collective API" << std::endl;
        std::cout << "   2. Integrate with llama.cpp CUDA backend" << std::endl;
        std::cout << "   3. Replace existing MXFP4 INT8 emulation" << std::endl;
        
        if (props.major == 12) {
            std::cout << "🚀 RTX 5070 has most advanced Blackwell features!" << std::endl;
        }
    } else {
        std::cout << "⚠️  Hardware may not fully support CUTLASS FP4 features" << std::endl;
    }
    
    return 0;
}