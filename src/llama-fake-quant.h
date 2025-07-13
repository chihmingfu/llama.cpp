#pragma once

#include "ggml.h"
#include "llama.h"

#ifdef __cplusplus
extern "C" {
#endif

// Fake quantization parameters
struct llama_fake_quant_params {
    bool enabled;                 // enable fake quantization
    enum ggml_type target_type;   // target quantization type (e.g., GGML_TYPE_BF16)
    float scale_factor;           // quantization scale factor (0.0-1.0, 1.0=all layers)
    bool compare_mode;            // output comparison between original and fake quantized
};

// Apply fake quantization to a single tensor
void llama_fake_quantize_tensor(
    struct ggml_tensor * tensor,
    enum ggml_type target_type,
    float scale_factor
);

// Apply fake quantization to tensor data (in-place)
void llama_fake_quantize_data(
    float * data,
    size_t n_elements,
    enum ggml_type target_type
);

// Apply fake quantization during inference pipeline
void llama_apply_fake_quantization(
    struct ggml_context * ctx,
    struct ggml_tensor * tensor,
    const struct llama_fake_quant_params * params
);

// Utility functions
const char * llama_fake_quant_type_name(enum ggml_type type);
bool llama_fake_quant_type_supported(enum ggml_type type);

#ifdef __cplusplus
}
#endif