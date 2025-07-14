#include "llama-fake-quant.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml.h"

#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>

// Apply fake quantization to tensor data (in-place)
void llama_fake_quantize_data(float * data, size_t n_elements, enum ggml_type target_type) {
    if (data == nullptr || n_elements == 0) {
        return;
    }

    switch (target_type) {
        case GGML_TYPE_BF16: {
            // Apply BF16 fake quantization: FP32 -> BF16 -> FP32
            for (size_t i = 0; i < n_elements; i++) {
                ggml_bf16_t bf16_val = ggml_compute_fp32_to_bf16(data[i]);
                data[i] = ggml_compute_bf16_to_fp32(bf16_val);
            }
            break;
        }
        case GGML_TYPE_F16: {
            // Apply FP16 fake quantization: FP32 -> FP16 -> FP32
            for (size_t i = 0; i < n_elements; i++) {
                ggml_fp16_t fp16_val = ggml_compute_fp32_to_fp16(data[i]);
                data[i] = ggml_compute_fp16_to_fp32(fp16_val);
            }
            break;
        }
        case GGML_TYPE_F32: {
            // No quantization needed for F32
            break;
        }
        default: {
            // For other quantization types, we would need to implement
            // temporary quantization and dequantization
            GGML_ABORT("Fake quantization for type %s not implemented yet", ggml_type_name(target_type));
            break;
        }
    }
}

// Apply fake quantization to a single tensor
void llama_fake_quantize_tensor(
    struct ggml_tensor * tensor,
    enum ggml_type target_type,
    float scale_factor
) {
    if (tensor == nullptr || tensor->data == nullptr) {
        return;
    }

    // Only apply to float tensors
    if (tensor->type != GGML_TYPE_F32) {
        return;
    }

    // Apply scale factor (randomly skip some elements based on scale_factor)
    const size_t n_elements = ggml_nelements(tensor);
    float * data = (float *)tensor->data;

    if (scale_factor >= 1.0f) {
        // Apply to all elements
        llama_fake_quantize_data(data, n_elements, target_type);
    } else if (scale_factor > 0.0f) {
        // Apply to a subset of elements (simple approach: apply to first scale_factor * n_elements)
        const size_t n_quantized = (size_t)(scale_factor * n_elements);
        llama_fake_quantize_data(data, n_quantized, target_type);
    }
    // If scale_factor <= 0.0f, no quantization is applied
}

// Apply fake quantization during inference pipeline
void llama_apply_fake_quantization(
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    const struct llama_fake_quant_params * params
) {
    (void)ctx; // Mark unused parameter
    if (params == nullptr || !params->enabled || tensor == nullptr) {
        return;
    }

    // Apply fake quantization to the tensor
    llama_fake_quantize_tensor(tensor, params->target_type, params->scale_factor);
}

// Utility functions
const char * llama_fake_quant_type_name(enum ggml_type type) {
    return ggml_type_name(type);
}

bool llama_fake_quant_type_supported(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            return true;
        default:
            return false;
    }
}

// Callback function for FFN norm fake quantization during inference
bool llama_ffn_norm_fake_quant_callback(struct ggml_tensor * tensor, bool ask, void * user_data) {
    if (!user_data) {
        return true;
    }
    
    struct llama_ffn_norm_fake_quant_data * data = (struct llama_ffn_norm_fake_quant_data *)user_data;
    
    if (!data->enabled) {
        return true;
    }
    
    // Check if this is an FFN norm tensor
    if (!tensor->name || strstr(tensor->name, "ffn_norm") == nullptr) {
        return true;
    }
    
    if (ask) {
        // Return true if we want to process this tensor
        return true;
    }
    
    // Extract layer number from tensor name (e.g., "blk.21.ffn_norm")
    int layer_num = -1;
    if (tensor->name) {
        char layer_str[32];
        if (sscanf(tensor->name, "blk.%d.ffn_norm", &layer_num) != 1) {
            // Try alternative naming patterns if needed
            return true;
        }
    }
    
    // Check if this is the target layer
    if (data->target_layer != -1 && layer_num != data->target_layer) {
        return true;
    }
    
    // Apply fake quantization if tensor has F32 data
    if (tensor->type == GGML_TYPE_F32 && tensor->data != nullptr) {
        float* tensor_data = (float*)tensor->data;
        size_t n_elements = ggml_nelements(tensor);
        
        // Store original values for verification
        float original_first = tensor_data[0];
        float original_last = tensor_data[n_elements - 1];
        
        // Apply fake quantization
        llama_fake_quantize_data(tensor_data, n_elements, data->target_type);
        
        // Verify quantization was applied
        float quantized_first = tensor_data[0];
        float quantized_last = tensor_data[n_elements - 1];
        
        printf("FFN norm fake quantization applied to %s (layer %d):\n", tensor->name, layer_num);
        printf("  Elements: %zu, Type: %s\n", n_elements, ggml_type_name(tensor->type));
        printf("  First value: %.8f -> %.8f (diff: %.8f)\n", 
               original_first, quantized_first, fabsf(original_first - quantized_first));
        printf("  Last value:  %.8f -> %.8f (diff: %.8f)\n", 
               original_last, quantized_last, fabsf(original_last - quantized_last));
               
        // Verify BF16 conversion is working
        ggml_bf16_t bf16_test = ggml_compute_fp32_to_bf16(original_first);
        float bf16_converted = ggml_compute_bf16_to_fp32(bf16_test);
        printf("  BF16 verification: %.8f -> %.8f (expected)\n", original_first, bf16_converted);
    } else {
        printf("FFN norm tensor %s skipped: type=%s, data=%p\n", 
               tensor->name ? tensor->name : "unknown", 
               ggml_type_name(tensor->type), 
               tensor->data);
    }
    
    return true;
}

// Global fake quantization state for GGML-level operations
static struct {
    bool enabled;
    enum ggml_type target_type;
    int target_layer;
} g_fake_quant_state = {false, GGML_TYPE_F32, -1};

// Set global fake quantization parameters for GGML-level operations
void llama_fake_quant_set_global_params(
    bool enabled,
    enum ggml_type target_type,
    int target_layer
) {
    g_fake_quant_state.enabled = enabled;
    g_fake_quant_state.target_type = target_type;
    g_fake_quant_state.target_layer = target_layer;
    
    if (enabled) {
        printf("GGML fake quantization enabled: type=%s, target_layer=%d\n", 
               ggml_type_name(target_type), target_layer);
    }
}

// Clear global fake quantization parameters
void llama_fake_quant_clear_global_params(void) {
    g_fake_quant_state.enabled = false;
    g_fake_quant_state.target_type = GGML_TYPE_F32;
    g_fake_quant_state.target_layer = -1;
}

// Check if fake quantization should be applied to RMS norm result
bool llama_fake_quant_should_apply_rms_norm(const struct ggml_tensor * tensor) {
    if (!g_fake_quant_state.enabled || !tensor || !tensor->name) {
        return false;
    }
    
    // Check if this is an FFN norm tensor
    if (strstr(tensor->name, "ffn_norm") == nullptr) {
        return false;
    }
    
    // Extract layer number from tensor name (e.g., "blk.21.ffn_norm")
    int layer_num = -1;
    if (sscanf(tensor->name, "blk.%d.ffn_norm", &layer_num) != 1) {
        return false;
    }
    
    // Check if this is the target layer
    return (g_fake_quant_state.target_layer == -1 || layer_num == g_fake_quant_state.target_layer);
}

// Apply fake quantization to RMS norm computation result
void llama_fake_quant_apply_rms_norm_result(
    float * data,
    size_t n_elements,
    const char * tensor_name
) {
    if (!g_fake_quant_state.enabled || !data || n_elements == 0) {
        return;
    }
    
    // Store original values for verification
    float original_first = data[0];
    float original_last = data[n_elements - 1];
    
    // Apply fake quantization
    llama_fake_quantize_data(data, n_elements, g_fake_quant_state.target_type);
    
    // Verify quantization was applied
    float quantized_first = data[0];
    float quantized_last = data[n_elements - 1];
    
    printf("GGML RMS norm fake quantization applied to %s:\n", tensor_name ? tensor_name : "unknown");
    printf("  Elements: %zu, Type: %s\n", n_elements, ggml_type_name(g_fake_quant_state.target_type));
    printf("  First value: %.8f -> %.8f (diff: %.8f)\n", 
           original_first, quantized_first, fabsf(original_first - quantized_first));
    printf("  Last value:  %.8f -> %.8f (diff: %.8f)\n", 
           original_last, quantized_last, fabsf(original_last - quantized_last));
           
    // Verify conversion is working
    ggml_bf16_t bf16_test = ggml_compute_fp32_to_bf16(original_first);
    float bf16_converted = ggml_compute_bf16_to_fp32(bf16_test);
    printf("  BF16 verification: %.8f -> %.8f (expected)\n", original_first, bf16_converted);
}