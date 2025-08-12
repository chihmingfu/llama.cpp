# CUTLASS 4.1 設置指南

本目錄需要 NVIDIA CUTLASS 4.1 庫以啟用 RTX 5070 Blackwell FP4 Tensor Core 加速。

## 快速設置

```bash
# 1. 進入 vendor 目錄
cd vendor/

# 2. 克隆 CUTLASS 4.1
git clone https://github.com/NVIDIA/cutlass.git
cd cutlass
git checkout v3.4.1  # 或最新的 4.x 分支

# 3. 返回 llama.cpp 根目錄並編譯
cd ../../
cmake -B build -DGGML_CUDA=ON -DGGML_CUDA_CUTLASS_FP4=ON
cmake --build build --config Release -j $(nproc)
```

## 使用 MXFP4 FP4 加速

```bash
# 啟用 CUTLASS FP4 路徑 (RTX 5070+ Blackwell)
GGML_CUDA_CUTLASS_FP4=1 ./build/bin/llama-bench -m model.mxfp4.gguf -p 512 -n 128 -ngl 999

# 使用 INT8 DP4A 回退路徑
GGML_CUDA_CUTLASS_FP4=0 ./build/bin/llama-bench -m model.mxfp4.gguf -p 512 -n 128 -ngl 999
```

## 硬體要求

- **GPU**: RTX 5070 或更新的 Blackwell 架構 (SM 12.0+)
- **CUDA**: 13.0 或更新版本
- **驅動**: R580 系列或更新

## 性能預期

基於 Llama 3.2 1B 模型測試：

| 配置 | 文字生成速度 | vs Q4_0 提升 |
|------|-------------|-------------|
| MXFP4 CUTLASS FP4 | 207.43 t/s | +115% |
| MXFP4 INT8 DP4A | 107.59 t/s | +12% |
| Q4_0 標準 | 96.45 t/s | 基線 |

## 故障排除

如果遇到編譯問題：

1. **確認 CUTLASS 版本**：使用 v3.4.1 或更新版本
2. **檢查 CUDA 版本**：需要 CUDA 13.0+
3. **驗證 GPU 架構**：確認是 SM 12.0 (Blackwell)

```bash
# 檢查 CUDA 版本
nvcc --version

# 檢查 GPU 架構
nvidia-smi
```

## 備選方案

如果無法設置 CUTLASS，MXFP4 仍可通過 INT8 DP4A 路徑正常工作：

```bash
# 不依賴 CUTLASS 的編譯
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j $(nproc)

# 自動使用 INT8 DP4A 路徑
./build/bin/llama-bench -m model.mxfp4.gguf -p 512 -n 128 -ngl 999
```

---

**更新**: 2025-01-12  
**相關報告**: claude_reports/021_blackwell_mxfp4_tensor_core_optimization_report.md