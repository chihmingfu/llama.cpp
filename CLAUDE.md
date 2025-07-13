# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

**Primary build system is CMake (Makefile is deprecated):**
```bash
# Standard CPU build
cmake -B build
cmake --build build --config Release

# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

# Multi-threaded build (faster)
cmake --build build --config Release -j $(nproc)

# Static build
cmake -B build -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release
```

**GPU/Accelerated builds:**
```bash
# CUDA build
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release

# Vulkan build
cmake -B build -DGGML_VULKAN=ON  
cmake --build build --config Release

# OpenCL build
cmake -B build -DGGML_OPENCL=ON
cmake --build build --config Release

# Metal build (macOS - enabled by default)
cmake -B build -DGGML_METAL=OFF  # to disable

# SYCL build (Intel GPU)
cmake -B build -DGGML_SYCL=ON
cmake --build build --config Release
```

**Common build issues and solutions:**
```bash
# If CURL not found, build without HTTP features
cmake -B build -DLLAMA_CURL=OFF
cmake --build build --config Release

# Clean build environment (WSL/cross-platform issues)
export PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
rm -rf build && cmake -B build
cmake --build build --config Release

# Specify CURL manually if detected incorrectly
cmake -B build -DCURL_LIBRARY=/usr/lib/x86_64-linux-gnu/libcurl.so.4
cmake --build build --config Release
```

**Development builds:**
```bash
# Build with tests
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release

# Build with sanitizers (debug builds)
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DLLAMA_SANITIZE_ADDRESS=ON
cmake --build build

# Build with fatal warnings
cmake -B build -DLLAMA_FATAL_WARNINGS=ON
cmake --build build --config Release

# Install built binaries to system (optional)
cmake --install build --prefix /usr/local
```

## Testing

**C++ unit tests:**
```bash
# Build tests
cmake -B build -DLLAMA_BUILD_TESTS=ON
cmake --build build --config Release

# Run specific tests
./build/bin/test-tokenizer-0
./build/bin/test-grammar-parser
./build/bin/test-sampling
./build/bin/test-quantize-fns
./build/bin/test-llama-grammar
```

**Server tests:**
```bash
cd tools/server/tests
./tests.sh
```

**Tool-specific tests:**
```bash
# GGUF split tests
cd tools/gguf-split && ./tests.sh

# Quantization tests  
cd tools/quantize && ./tests.sh

# MTMD tests
cd tools/mtmd && ./tests.sh

# Server integration tests (Python-based)
cd tools/server/tests && ./tests.sh
```

## Code Quality Tools

**Formatting and Linting:**
```bash
# C++ formatting (clang-format)
clang-format -i src/**/*.cpp src/**/*.h

# C++ static analysis (clang-tidy)  
clang-tidy src/**/*.cpp

# Python formatting and linting
python -m flake8

# Pre-commit hooks
pre-commit run --all-files
```

## Key Tools and Binaries

All tools are built to `build/bin/` directory:

**Core Tools:**
- `llama-cli` - Main command line interface for inference (tools/main/)
- `llama-server` - OpenAI-compatible HTTP server (tools/server/)
- `llama-run` - Interactive CLI with readline support (tools/run/)

**Benchmarking and Analysis:**
- `llama-bench` - Performance benchmarking (tools/llama-bench/)
- `llama-batched-bench` - Batched inference benchmarking (tools/batched-bench/)
- `llama-perplexity` - Model quality evaluation (tools/perplexity/)

**Model Tools:**
- `llama-quantize` - Model quantization (tools/quantize/)
- `llama-imatrix` - Importance matrix generation for quantization (tools/imatrix/)
- `llama-export-lora` - Export LoRA adapters (tools/export-lora/)

**Utilities:**
- `llama-gguf-split` - Split large GGUF files (tools/gguf-split/)
- `llama-tokenize` - Tokenization analysis (tools/tokenize/)
- `llama-cvector-generator` - Control vector generation (tools/cvector-generator/)
- `llama-rpc-server` - RPC server for distributed inference (tools/rpc/)

**Multimodal:**
- `llama-mtmd-cli` - Multimodal support (vision/audio) - replaces legacy llava-cli, minicpmv-cli, etc. (tools/mtmd/)
- `llama-tts` - Text-to-speech functionality (tools/tts/)

## Architecture Overview

**Core library structure:**
- `src/` - Main llama.cpp library implementation
- `include/llama.h` - C-style API interface
- `ggml/` - GGML tensor operations library (git submodule)
- `common/` - Shared utilities across tools

**Key components:**
- `llama-model.cpp` - Model loading and architecture definitions
- `llama-context.cpp` - Inference context management  
- `llama-sampling.cpp` - Text generation sampling strategies
- `llama-kv-cache-unified*.cpp` - Key-value cache implementations
- `llama-vocab.cpp` - Tokenizer implementations
- `llama-arch.cpp` - Architecture-specific implementations
- `llama-model-loader.cpp` - Model file loading and validation
- `llama-memory*.cpp` - Memory management for different cache types
- `llama-grammar.cpp` - Grammar-guided generation support
- `llama-chat.cpp` - Chat template processing

**Tools architecture:**
- `tools/main/` - llama-cli implementation
- `tools/server/` - HTTP server with OpenAI API compatibility
- `tools/mtmd/` - Multimodal (vision/audio) support
- `tools/quantize/` - Model quantization utilities
- Each tool has its own CMakeLists.txt and README.md

**Examples structure:**
- `examples/` - Comprehensive example applications
- `examples/simple/` - Minimal usage examples
- `examples/server/` - Server usage examples
- Each example has CMakeLists.txt for standalone building

**Testing structure:**
- `tests/` - C++ unit tests
- `tools/*/tests/` - Tool-specific integration tests
- GitHub Actions workflows in `.github/workflows/`

**Model support:**
- Supports 100+ model architectures (LLaMA, Mistral, Qwen, etc.)
- GGUF format for quantized models
- Multimodal support for vision and audio models

**Backend support:**
- CPU with SIMD optimizations (AVX, NEON)
- NVIDIA GPU via CUDA
- AMD GPU via HIP  
- Intel GPU via SYCL
- Apple Silicon via Metal
- Vulkan cross-platform GPU support
- OpenCL support

## Recent Important Changes

**Breaking changes and deprecations:**
- `llama-mtmd-cli` replaces legacy multimodal tools: `llava-cli`, `minicpmv-cli`, `gemma3-cli`, `qwen2vl-cli`
- `libllava` is deprecated in favor of unified multimodal support in `tools/mtmd/`
- Server now has universal tool call support and multimodal capabilities
- Refer to API changelogs: [libllama API](https://github.com/ggml-org/llama.cpp/issues/9289), [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

**New features:**
- Hugging Face direct model download support (`-hf` flag)
- Multimodal support in `llama-server`
- Universal function calling support
- GGUF-my-LoRA for LoRA conversion and merging

## Development Workflow

1. **Building from source:** Use CMake as primary build system
2. **Code formatting:** Use clang-format and pre-commit hooks
3. **Adding new models:** Follow `docs/development/HOWTO-add-model.md`
4. **Performance optimization:** See `docs/development/token_generation_performance_tips.md`
5. **Testing:** Run relevant test suites before submitting changes
6. **Python linting:** Use flake8 for Python scripts

## Dependencies

**Build dependencies:**
- C++17 compiler required
- CMake 3.14 or higher
- libcurl development headers (for HTTP server and HuggingFace downloads)
- Optional: CUDA toolkit, Intel oneAPI, OpenBLAS, Vulkan SDK

**Python dependencies:**
- Python scripts use dependencies in `requirements/` directory
- Poetry configuration in `pyproject.toml`
- Core conversion scripts: `requirements-convert_hf_to_gguf.txt`

**Development dependencies:**
- clang-format for C++ formatting
- clang-tidy for static analysis
- pre-commit for automated quality checks
- flake8 for Python linting

**Runtime:**
- No major external dependencies for core library (self-contained)

## Important Implementation Details

**Core Architecture Flow:**
- Model loading: `llama-model-loader.cpp` → `llama-model.cpp` → `llama-arch.cpp` (architecture-specific)
- Inference pipeline: `llama-context.cpp` manages state, `llama-sampling.cpp` handles generation
- Memory management: Multiple cache types via `llama-memory*.cpp` and unified KV cache
- Backend dispatch: GGML handles compute backend selection (CPU/CUDA/Metal/etc.)

**Key Configuration Points:**
- Model architecture detection in `llama-arch.cpp` with 100+ supported model types
- Context parameters in `llama-cparams.cpp` control memory and performance
- Hyperparameters in `llama-hparams.cpp` define model-specific settings
- Vocabulary handling in `llama-vocab.cpp` supports multiple tokenizer types

**Thread Safety and Performance:**
- Core library is thread-safe for concurrent inference contexts
- Batch processing supported via `llama-batch.cpp` for efficient parallel generation
- Memory-mapped model loading via `llama-mmap.cpp` for fast startup and memory efficiency
- SIMD optimizations handled automatically by GGML backend

## Debugging and Troubleshooting

**Common Debug Workflows:**
```bash
# Debug build with verbose output
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DLLAMA_DEBUG=ON
cmake --build build

# Run with debug logging
LLAMA_LOG_LEVEL=debug ./build/bin/llama-cli -m model.gguf

# Memory debugging
cmake -B build -DCMAKE_BUILD_TYPE=Debug -DLLAMA_SANITIZE_ADDRESS=ON
cmake --build build
```

**Performance Analysis:**
- Use `llama-bench` for standardized performance testing
- Monitor GPU utilization with `nvidia-smi` (CUDA) or `rocm-smi` (ROCm)
- Profile with `perf`, `gprof`, or platform-specific tools
- Check model compatibility with target hardware before optimization

**Model Conversion Issues:**
- Use `convert_hf_to_gguf.py` for Hugging Face models
- Verify model architecture support in `src/llama-arch.cpp`
- Check tokenizer compatibility in conversion process
- Test converted models with simple inference before full deployment