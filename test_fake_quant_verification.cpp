#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include "ggml.h"

// Forward declare our fake quantization functions
extern "C" {
    void ggml_fake_quant_set_global_params(bool enabled, int target_type, int target_layer);
}

// BF16 conversion functions for testing
ggml_bf16_t test_fp32_to_bf16(float x) {
    ggml_bf16_t bf16_val = ggml_fp32_to_bf16(x);
    return bf16_val;
}

float test_bf16_to_fp32(ggml_bf16_t x) {
    return ggml_bf16_to_fp32(x);
}

// Test BF16 fake quantization directly
void test_bf16_quantization() {
    std::cout << "=== BF16 Fake Quantization Unit Test ===" << std::endl;
    
    // Test values that should show precision loss
    std::vector<float> test_values = {
        1.23456789f,
        -0.987654321f,
        0.000123456f,
        -1000.123456f,
        3.14159265359f
    };
    
    for (float original : test_values) {
        ggml_bf16_t bf16_val = test_fp32_to_bf16(original);
        float quantized = test_bf16_to_fp32(bf16_val);
        float diff = fabs(original - quantized);
        
        std::cout << "Original: " << original 
                  << " -> BF16: " << quantized 
                  << " (diff: " << diff << ")" << std::endl;
        
        if (diff < 1e-8) {
            std::cout << "  WARNING: No precision loss detected!" << std::endl;
        }
    }
}

// Test tensor name pattern matching
void test_tensor_name_patterns() {
    std::cout << "\n=== Tensor Name Pattern Test ===" << std::endl;
    
    std::vector<std::string> test_names = {
        "norm-0",
        "norm-21", 
        "blk.5.ffn_norm",
        "blk.10.attn_norm",
        "output_norm",
        "some_other_tensor"
    };
    
    for (const std::string& name : test_names) {
        int layer_num = -1;
        bool matches_norm_pattern = (sscanf(name.c_str(), "norm-%d", &layer_num) == 1);
        bool matches_ffn_pattern = (name.find("ffn_norm") != std::string::npos);
        
        std::cout << "Tensor: " << name;
        if (matches_norm_pattern) {
            std::cout << " -> norm-pattern, layer=" << layer_num;
        } else if (matches_ffn_pattern) {
            std::cout << " -> ffn_norm pattern";
        } else {
            std::cout << " -> no match";
        }
        std::cout << std::endl;
    }
}

// Test array quantization with verification
void test_array_quantization() {
    std::cout << "\n=== Array Quantization Test ===" << std::endl;
    
    const size_t array_size = 100;
    std::vector<float> data(array_size);
    
    // Fill with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
    
    for (size_t i = 0; i < array_size; i++) {
        data[i] = dis(gen);
    }
    
    // Store original values
    std::vector<float> original_data = data;
    
    // Apply BF16 fake quantization
    size_t quantized_count = 0;
    for (size_t i = 0; i < array_size; i++) {
        ggml_bf16_t bf16_val = test_fp32_to_bf16(data[i]);
        float quantized = test_bf16_to_fp32(bf16_val);
        
        if (fabs(data[i] - quantized) > 1e-8) {
            quantized_count++;
        }
        
        data[i] = quantized;
    }
    
    std::cout << "Quantized " << quantized_count << "/" << array_size 
              << " elements (" << (100.0 * quantized_count / array_size) << "%)" << std::endl;
    
    // Calculate statistics
    float max_diff = 0.0f;
    float avg_diff = 0.0f;
    for (size_t i = 0; i < array_size; i++) {
        float diff = fabs(original_data[i] - data[i]);
        max_diff = std::max(max_diff, diff);
        avg_diff += diff;
    }
    avg_diff /= array_size;
    
    std::cout << "Max difference: " << max_diff << std::endl;
    std::cout << "Avg difference: " << avg_diff << std::endl;
    
    if (quantized_count == 0) {
        std::cout << "ERROR: No quantization effects detected!" << std::endl;
    } else {
        std::cout << "SUCCESS: Quantization effects confirmed" << std::endl;
    }
}

int main() {
    std::cout << "Fake Quantization Verification Test Suite" << std::endl;
    std::cout << "=========================================" << std::endl;
    
    test_bf16_quantization();
    test_tensor_name_patterns();
    test_array_quantization();
    
    std::cout << "\n=== Test Complete ===" << std::endl;
    return 0;
}