# MXFP4 量化支援說明

## 概述
llama.cpp 最近新增了對 MXFP4 (Microscaling FP4) 格式的支援。這是與 NVIDIA 合作開發的新量化格式，特別針對 OpenAI 的 gpt-oss 模型優化。

## 目前支援狀況

### 1. MXFP4 量化選項
根據 `llama-quantize` 工具，目前支援：
- **MXFP4_MOE**: 專門為 MoE (Mixture of Experts) 模型設計的 MXFP4 量化

### 2. 相關資訊
- **PR**: [#15091](https://github.com/ggml-org/llama.cpp/pull/15091)
- **合作方**: NVIDIA
- **主要應用**: gpt-oss 模型的原生 MXFP4 格式支援
- **討論**: [#15095](https://github.com/ggml-org/llama.cpp/discussions/15095)

## 使用限制

### 目前的限制：
1. **模型支援有限**: 主要為 gpt-oss 模型設計，不是所有模型都適合使用 MXFP4
2. **MoE 專用**: 目前的 MXFP4_MOE 選項似乎專門為 MoE 架構優化
3. **實驗性功能**: 這是較新的功能，可能還在持續開發中

## 如何使用 MXFP4 量化

### 對於一般模型（如 Llama 3.2）
```bash
# 注意：Llama 3.2 1B 不是 MoE 模型，MXFP4_MOE 可能不適用
# 但可以嘗試：
./build/bin/llama-quantize ./models/llama-3.2-1b-f16.gguf \
  ./models/llama-3.2-1b-mxfp4.gguf MXFP4_MOE
```

### 建議的替代方案
對於 Llama 3.2 1B 這樣的非 MoE 模型，建議使用：
1. **Q4_K_M**: 4-bit K-quants，品質和大小的良好平衡
2. **Q4_0/Q4_1**: 標準 4-bit 量化
3. **Q5_K_M**: 5-bit 量化，品質更好

## MXFP4 技術背景

### 什麼是 MXFP4？
- **Microscaling FP4**: 一種新的 4-bit 浮點格式
- **特點**: 
  - 使用共享的縮放因子（microscaling）
  - 比傳統 INT4 保留更多動態範圍
  - 特別適合 Transformer 模型

### 優勢：
1. **更好的精度**: 相比傳統 INT4 量化
2. **硬體優化**: 特別是在支援的 NVIDIA GPU 上
3. **動態範圍**: 保持浮點數的動態範圍特性

### 劣勢：
1. **硬體需求**: 可能需要特定硬體支援才能發揮優勢
2. **相容性**: 目前支援的模型有限
3. **實驗性**: 還在早期階段

## 實驗建議

### 1. 檢查模型相容性
```bash
# 嘗試量化並觀察錯誤訊息
./build/bin/llama-quantize ./models/your-model-f16.gguf \
  ./models/test-mxfp4.gguf MXFP4_MOE
```

### 2. 比較測試
如果量化成功，比較：
- 檔案大小
- 推理速度
- 模型品質（困惑度）

### 3. 等待更多支援
MXFP4 是新功能，建議：
- 關注 llama.cpp 的更新
- 查看是否有針對一般模型的 MXFP4 選項（非 MoE）
- 追蹤相關 GitHub 討論和 PR

## 結論

MXFP4 是一個有潛力的新量化格式，但目前在 llama.cpp 中的支援還比較有限，主要針對特定的 MoE 模型。對於 Llama 3.2 1B 這樣的標準模型，建議暫時使用成熟的量化格式如 Q4_K_M 或 Q5_K_M。

隨著開發的進展，未來可能會有更廣泛的 MXFP4 支援。