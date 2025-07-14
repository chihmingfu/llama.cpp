#include <stdio.h>
#include <math.h>
#include <stdint.h>

// Simple BF16 implementation for testing
typedef uint16_t bf16_t;

bf16_t fp32_to_bf16(float x) {
    union { float f; uint32_t i; } u = { x };
    return (bf16_t)(u.i >> 16);
}

float bf16_to_fp32(bf16_t x) {
    union { float f; uint32_t i; } u = { 0 };
    u.i = ((uint32_t)x) << 16;
    return u.f;
}

int main() {
    printf("BF16 Quantization Test\n");
    printf("======================\n");
    
    float test_values[] = {
        1.23456789f,
        -0.987654321f, 
        0.000123456f,
        -1000.123456f,
        3.14159265359f
    };
    
    int num_tests = sizeof(test_values) / sizeof(test_values[0]);
    int quantized_count = 0;
    
    for (int i = 0; i < num_tests; i++) {
        float original = test_values[i];
        bf16_t bf16_val = fp32_to_bf16(original);
        float quantized = bf16_to_fp32(bf16_val);
        float diff = fabsf(original - quantized);
        
        printf("Test %d: %.8f -> %.8f (diff: %.8f)\n", 
               i+1, original, quantized, diff);
        
        if (diff > 1e-8) {
            quantized_count++;
            printf("  ✓ Quantization effect detected\n");
        } else {
            printf("  ✗ No quantization effect\n");
        }
    }
    
    printf("\nSummary: %d/%d values showed quantization effects\n", 
           quantized_count, num_tests);
    
    if (quantized_count > 0) {
        printf("SUCCESS: BF16 quantization is working\n");
        return 0;
    } else {
        printf("ERROR: BF16 quantization not working\n"); 
        return 1;
    }
}