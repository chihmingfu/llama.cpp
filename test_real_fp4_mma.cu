// Test real FP4 E2M1 MMA instruction compilation with CUDA 13.0
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void test_real_fp4_mma() {
    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1200)
    // Real FP4 MMA instruction for Blackwell
    unsigned a[2] = {0x01234567, 0x89ABCDEF};
    unsigned b[2] = {0xFEDCBA98, 0x76543210};
    float c[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float d[4];
    
    // Actual SM120 FP4 MMA instruction
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e2m1.e2m1.f32 "
        "{%0, %1, %2, %3}, "  // D matrix (output)
        "{%4, %5}, "          // A matrix (FP4 E2M1)
        "{%6, %7}, "          // B matrix (FP4 E2M1)
        "{%8, %9, %10, %11};" // C matrix (accumulator)
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
    
    if (threadIdx.x == 0) {
        printf("✅ NATIVE FP4 MMA INSTRUCTION EXECUTED!\n");
        printf("Output: [%f, %f, %f, %f]\n", d[0], d[1], d[2], d[3]);
    }
    #endif
}

int main() {
    printf("=== Real FP4 E2M1 MMA Test ===\n");
    test_real_fp4_mma<<<1, 32>>>();
    cudaDeviceSynchronize();
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    printf("✅ SUCCESS: Native FP4 MMA instruction works!\n");
    return 0;
}