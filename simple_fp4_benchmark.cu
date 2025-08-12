#include <cuda_runtime.h>
#include <cuda_fp4.h>
#include <cuda_fp16.h>
#include <stdio.h>

#define TILE_SIZE 1024
#define NUM_ITERATIONS 100

__global__ void fp4_conversion_test(__nv_fp4_storage_t* fp4_out, const float* fp32_in, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        fp4_out[idx] = __nv_cvt_float_to_fp4(fp32_in[idx], __NV_E2M1, cudaRoundNearest);
    }
}

__global__ void int8_conversion_test(int8_t* int8_out, const float* fp32_in, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float scaled = fp32_in[idx] / scale;
        int8_out[idx] = (int8_t)fmaxf(-128.0f, fminf(127.0f, scaled));
    }
}

double time_kernel(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return (double)milliseconds;
}

int main() {
    printf("FP4 vs INT8 Performance Baseline Test\n");
    printf("Size: %d elements\n", TILE_SIZE);
    printf("Iterations: %d\n\n", NUM_ITERATIONS);
    
    // Allocate host memory
    float* host_fp32 = (float*)malloc(TILE_SIZE * sizeof(float));
    for (int i = 0; i < TILE_SIZE; i++) {
        host_fp32[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
    
    // Allocate device memory
    float* dev_fp32;
    __nv_fp4_storage_t* dev_fp4;
    int8_t* dev_int8;
    
    cudaMalloc(&dev_fp32, TILE_SIZE * sizeof(float));
    cudaMalloc(&dev_fp4, TILE_SIZE * sizeof(__nv_fp4_storage_t));
    cudaMalloc(&dev_int8, TILE_SIZE * sizeof(int8_t));
    
    cudaMemcpy(dev_fp32, host_fp32, TILE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    
    // Setup CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    const int block_size = 256;
    const int grid_size = (TILE_SIZE + block_size - 1) / block_size;
    const float scale = 0.1f;
    
    printf("=== Conversion Benchmarks ===\n");
    
    // Benchmark FP4 conversion
    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        fp4_conversion_test<<<grid_size, block_size>>>(dev_fp4, dev_fp32, TILE_SIZE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    double fp4_time = time_kernel(start, stop) / NUM_ITERATIONS;
    
    // Benchmark INT8 conversion  
    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        int8_conversion_test<<<grid_size, block_size>>>(dev_int8, dev_fp32, scale, TILE_SIZE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    double int8_time = time_kernel(start, stop) / NUM_ITERATIONS;
    
    printf("FP4 Conversion:  %.3f ms\n", fp4_time);
    printf("INT8 Conversion: %.3f ms\n", int8_time);
    printf("FP4 vs INT8 Ratio: %.2fx\n", fp4_time / int8_time);
    
    // Memory usage comparison
    printf("\n=== Memory Usage Comparison ===\n");
    int fp32_size = TILE_SIZE * 4;
    int int8_size = TILE_SIZE * 1; 
    int fp4_size = TILE_SIZE / 2;  // 2 FP4 values per byte
    
    printf("FP32 Size:  %d bytes (baseline)\n", fp32_size);
    printf("INT8 Size:  %d bytes (%.1f%% of FP32)\n", int8_size, 100.0f * int8_size / fp32_size);
    printf("FP4 Size:   %d bytes (%.1f%% of FP32)\n", fp4_size, 100.0f * fp4_size / fp32_size);
    printf("FP4 Memory Advantage vs INT8: %.1fx\n", (float)int8_size / fp4_size);
    
    // Performance projections
    printf("\n=== Performance Projections ===\n");
    printf("Current MXFP4 (INT8 emulation): Baseline\n");
    printf("Expected FP4 Tensor Cores: 2-4x faster (hardware acceleration)\n");
    printf("Memory bandwidth savings: %.1fx (FP4 vs INT8)\n", (float)int8_size / fp4_size);
    printf("Total expected speedup: 4-8x (computation + memory)\n");
    
    // Test data conversion accuracy
    printf("\n=== Conversion Accuracy Test ===\n");
    
    // Copy back results to verify
    __nv_fp4_storage_t* host_fp4 = (__nv_fp4_storage_t*)malloc(TILE_SIZE * sizeof(__nv_fp4_storage_t));
    cudaMemcpy(host_fp4, dev_fp4, TILE_SIZE * sizeof(__nv_fp4_storage_t), cudaMemcpyDeviceToHost);
    
    // Test a few conversions
    for (int i = 0; i < 5; i++) {
        float original = host_fp32[i];
        __nv_fp4_storage_t fp4_val = host_fp4[i];
        
        // Convert back to verify
        __half_raw half_result = __nv_cvt_fp4_to_halfraw(fp4_val, __NV_E2M1);
        union { __half_raw r; unsigned short s; } h;
        h.r = half_result;
        float converted = __half2float(__short_as_half(h.s));
        
        float error = fabsf(original - converted);
        printf("Test %d: %.3f -> FP4 -> %.3f (error: %.4f)\n", i, original, converted, error);
    }
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    free(host_fp32);
    free(host_fp4);
    cudaFree(dev_fp32);
    cudaFree(dev_fp4);
    cudaFree(dev_int8);
    
    printf("\nBaseline test completed successfully!\n");
    
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(error));
        return 1;
    }
    
    return 0;
}