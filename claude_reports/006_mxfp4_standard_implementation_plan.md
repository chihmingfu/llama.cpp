# MXFP4 標準量化實現計劃

## 執行摘要

本計劃提出為 llama.cpp 添加標準 MXFP4 量化支援，讓非 MoE 模型（如 Llama 3.2 1B）也能使用 MXFP4 4-bit 量化格式。目前的 MXFP4_MOE 實現會在標準模型上自動回退到 Q8_0，無法發揮 MXFP4 的壓縮優勢。

## 背景分析

### 現有問題
1. **MXFP4_MOE 的限制**：
   - 只對 `tensor->ne[2] > 1` 的張量使用 MXFP4（MoE 專家張量）
   - 標準模型所有張量都回退到 Q8_0
   - 無法實現預期的 4-bit 壓縮

2. **技術潛力**：
   - MXFP4 是 Microsoft FP4 格式，具有良好的動態範圍
   - 使用共享指數 + 4-bit mantissa
   - 理論上可達到約 4.25 bits/weight 的壓縮率

3. **實現基礎**：
   - GGML 已有完整的 MXFP4 基礎實現
   - CUDA、Metal、Vulkan 等後端已支援 MXFP4 運算
   - 只需修改量化策略邏輯

## 設計方案

### 方案一：新增 LLAMA_FTYPE_MOSTLY_MXFP4（推薦）

**優點**：
- 清晰區分 MoE 和標準用途
- 保持向後兼容性
- 符合現有命名規範

**實現要點**：
1. 新增 `LLAMA_FTYPE_MOSTLY_MXFP4 = 39` 枚舉值
2. 在量化邏輯中對所有適合的張量使用 MXFP4
3. 保留 1D 張量和 normalization 層的特殊處理

### 方案二：擴展現有 MXFP4_MOE 的行為

**優點**：
- 修改最小
- 不增加新的枚舉值

**缺點**：
- 破壞現有語義
- 可能影響已有 MoE 模型的量化行為
- 不建議採用

### 方案三：添加運行時參數控制

**優點**：
- 最大靈活性
- 用戶可選擇量化策略

**缺點**：
- 增加使用複雜度
- 需要修改命令行接口

## 詳細實現步驟

### 第一階段：核心實現（預計 2-3 小時）

#### 1. 修改枚舉定義
**文件**：`/workspace/llama.cpp/include/llama.h`
```cpp
// 在第 155 行後添加
LLAMA_FTYPE_MOSTLY_MXFP4 = 39,  // standard MXFP4, except 1d tensors
```

#### 2. 更新量化工具映射
**文件**：`/workspace/llama.cpp/tools/quantize/quantize.cpp`
```cpp
// 在 QUANT_OPTIONS 表中添加
{ "MXFP4",    LLAMA_FTYPE_MOSTLY_MXFP4,    " 4.25 bpw MXFP4 quantization", },
```

#### 3. 實現量化邏輯
**文件**：`/workspace/llama.cpp/src/llama-quant.cpp`

在第 229 行附近添加新的條件分支：
```cpp
} else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4) {
    // Standard MXFP4 quantization for all applicable tensors
    // Skip 1D tensors and normalization layers
    if (name.find("_norm") != std::string::npos ||
        tensor->ne[0] == tensor->nelements()) {
        // Keep F32 for normalization and 1D tensors
        new_type = cur_type;
    } else if (name == "token_embd.weight" || 
               name == "output.weight") {
        // Use Q6_K for embeddings and output for better quality
        new_type = GGML_TYPE_Q6_K;
    } else {
        // Use MXFP4 for all other tensors
        new_type = GGML_TYPE_MXFP4;
    }
} else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
    // Keep existing MoE behavior
    if (tensor->ne[2] > 1) {
        new_type = GGML_TYPE_MXFP4;
    } else {
        new_type = GGML_TYPE_Q8_0;
    }
}
```

#### 4. 更新模型加載器
**文件**：`/workspace/llama.cpp/src/llama-model-loader.cpp`

確保新的 ftype 被正確識別和處理。

### 第二階段：Python 支援（預計 1 小時）

#### 5. 更新 Python 常量
**文件**：`/workspace/llama.cpp/gguf-py/gguf/constants.py`
```python
# 添加新的 ftype 常量
class LlamaFileType(IntEnum):
    # ... existing entries ...
    MOSTLY_MXFP4 = 39  # Standard MXFP4
```

#### 6. 更新轉換腳本
**文件**：`/workspace/llama.cpp/convert_hf_to_gguf.py`

添加對新量化類型的支援。

### 第三階段：測試與驗證（預計 2-3 小時）

#### 7. 編譯測試
```bash
# 重新編譯
cmake --build build --config Release -j $(nproc)

# 驗證新選項
./build/bin/llama-quantize --help | grep MXFP4
```

#### 8. 量化測試
```bash
# 測試標準 MXFP4 量化
./build/bin/llama-quantize models/llama-3.2-1b.gguf \
  models/llama-3.2-1b-mxfp4.gguf MXFP4

# 驗證檔案大小（預期約 600-700MB）
ls -lh models/llama-3.2-1b-mxfp4.gguf
```

#### 9. 性能測試
```bash
# Benchmark 測試
./build/bin/llama-bench -m models/llama-3.2-1b-mxfp4.gguf -p 512 -n 128

# Perplexity 測試
./build/bin/llama-perplexity -m models/llama-3.2-1b-mxfp4.gguf -f wiki.test.txt
```

#### 10. 推理測試
```bash
# 實際推理測試
./build/bin/llama-cli -m models/llama-3.2-1b-mxfp4.gguf \
  -p "Once upon a time" -n 50 -ngl 99
```

## 技術考量

### 1. 張量選擇策略

**建議保持高精度的張量**：
- **Normalization 層**（`*_norm`）：保持 F32
- **Token embeddings**：使用 Q6_K 或更高
- **Output layer**：使用 Q6_K 或更高
- **1D 張量**：保持原始精度

**適合 MXFP4 的張量**：
- **Attention 權重**：Q、K、V、O 投影
- **FFN 權重**：up、down、gate 投影
- **其他 2D/3D 權重張量**

### 2. 品質與性能權衡

**預期指標**：
- **檔案大小**：約 600-700MB（比 Q4_0 小約 10%）
- **Perplexity**：預計在 Q4_0 和 Q4_K_M 之間
- **推理速度**：可能略快於 Q4_K_M
- **記憶體使用**：顯著降低

### 3. 硬體兼容性

**已支援的後端**：
- ✅ CPU (x86/ARM with SIMD)
- ✅ CUDA
- ✅ Metal
- ✅ Vulkan
- ⚠️ OpenCL（需要驗證）
- ⚠️ SYCL（需要驗證）

## 風險評估

### 技術風險

1. **品質風險**（中）：
   - MXFP4 在某些層可能導致品質下降
   - 緩解：選擇性量化策略，關鍵層使用更高精度

2. **兼容性風險**（低）：
   - 新增枚舉值可能影響下游工具
   - 緩解：保持向後兼容，充分測試

3. **性能風險**（低）：
   - MXFP4 解碼可能在某些硬體上較慢
   - 緩解：提供多種量化選項供用戶選擇

### 實施風險

1. **時間風險**（低）：
   - 實現相對簡單，主要是配置變更
   - 預計總時間：6-8 小時

2. **測試風險**（中）：
   - 需要全面測試各種模型
   - 緩解：制定詳細測試計劃

## 成功標準

1. **功能完成**：
   - ✅ 新量化選項在工具中可見
   - ✅ 標準模型能成功使用 MXFP4 量化
   - ✅ 生成的模型能正常推理

2. **性能指標**：
   - ✅ 檔案大小減少 > 60%（相對 F16）
   - ✅ Perplexity < 12.0（優於 Q4_0）
   - ✅ 推理速度 > 400 t/s（在 RTX 5070）

3. **品質保證**：
   - ✅ 通過現有單元測試
   - ✅ 無記憶體洩漏或崩潰
   - ✅ 向後兼容性保持

## 實施時間表

| 階段 | 任務 | 預計時間 | 依賴 |
|------|------|----------|------|
| 1 | 核心實現 | 2-3 小時 | - |
| 2 | Python 支援 | 1 小時 | 階段 1 |
| 3 | 編譯與基礎測試 | 1 小時 | 階段 2 |
| 4 | 性能與品質測試 | 2 小時 | 階段 3 |
| 5 | 文檔更新 | 1 小時 | 階段 4 |
| **總計** | | **7-8 小時** | |

## 後續優化建議

### 短期（1-2 週）
1. 添加 imatrix 支援以改善量化品質
2. 實現混合精度策略（不同層使用不同量化）
3. 優化 MXFP4 kernel 性能

### 中期（1-2 月）
1. 支援 MXFP6、MXFP8 等其他 Microscaling 格式
2. 開發自動量化策略選擇器
3. 整合到 llama-server 的動態量化

### 長期（3-6 月）
1. 研究 MXFP4 的訓練感知量化
2. 開發專用的 MXFP4 優化器
3. 探索硬體加速可能性

## 結論

添加標準 MXFP4 支援是一個有價值的增強，能為用戶提供更多量化選擇。實現相對簡單，風險可控，預期能在 4-bit 量化領域提供更好的品質-壓縮權衡。建議採用方案一（新增 LLAMA_FTYPE_MOSTLY_MXFP4），並按照上述計劃逐步實施。

---

**文檔版本**：1.0  
**創建日期**：2025-08-07  
**作者**：Claude Assistant  
**狀態**：待實施  
**預計工時**：7-8 小時