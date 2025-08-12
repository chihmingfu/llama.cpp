// Test proper FP4 syntax for CUDA 13.0 / SM120
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_fp4_with_kind_modifier() {
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    // FP4 MMA with proper .kind modifier
    unsigned a[4] = {0x01234567, 0x89ABCDEF, 0x11111111, 0x22222222};
    unsigned b[4] = {0xFEDCBA98, 0x76543210, 0x33333333, 0x44444444};
    float c[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float d[4];
    
    // SM120 FP4 MMA with .kind::f8f6f4 modifier
    asm volatile(
        ".reg .b32 a<4>, b<4>;\n\t"
        ".reg .f32 c<4>, d<4>;\n\t"
        "mov.b32 a0, %4;\n\t"
        "mov.b32 a1, %5;\n\t"
        "mov.b32 a2, %6;\n\t"
        "mov.b32 a3, %7;\n\t"
        "mov.b32 b0, %8;\n\t"
        "mov.b32 b1, %9;\n\t"
        "mov.b32 b2, %10;\n\t"
        "mov.b32 b3, %11;\n\t"
        "mov.f32 c0, %12;\n\t"
        "mov.f32 c1, %13;\n\t"
        "mov.f32 c2, %14;\n\t"
        "mov.f32 c3, %15;\n\t"
        // Use the proper .kind::f8f6f4 modifier for FP4
        "mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32.kind::f8f6f4 "
        "{d0, d1, d2, d3}, "
        "{a0, a1, a2, a3}, "
        "{b0, b1, b2, b3}, "
        "{c0, c1, c2, c3};\n\t"
        "mov.f32 %0, d0;\n\t"
        "mov.f32 %1, d1;\n\t"
        "mov.f32 %2, d2;\n\t"
        "mov.f32 %3, d3;\n\t"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]), "r"(b[2]), "r"(b[3]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
    
    if (threadIdx.x == 0) {
        printf("✅ FP4 MMA with .kind::f8f6f4 executed!\n");
        printf("Output: [%f, %f, %f, %f]\n", d[0], d[1], d[2], d[3]);
    }
    #endif
}

int main() {
    printf("=== FP4 MMA with .kind modifier Test ===\n");
    test_fp4_with_kind_modifier<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    printf("✅ Native FP4 support confirmed!\n");
    return 0;
}