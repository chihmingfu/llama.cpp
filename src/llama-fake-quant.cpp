#include "llama-fake-quant.h"
#include "../ggml/src/ggml-impl.h"
#include "ggml.h"

#include <cstring>
#include <cassert>

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