# FFN Norm BF16 Fake Quantization 實驗設計

## 🎯 實驗目標

測試在 FFN normalization 層應用 BF16 fake quantization 對模型推理精度的影響，評估比 logits 層面更早期的精度損失對整體模型性能的影響。

## 📍 FFN Norm 在模型中的位置分析

### Transformer 層結構：
```
Input → Attention → Add&Norm → **FFN_NORM** → FFN → Add&Norm → Output
                                    ↑
                               我們的目標位置
```

### 在 llama.cpp 中的實現位置：
- **文件**: `src/llama-model.cpp`
- **函數**: 各種架構的前向傳播函數
- **模式**: `build_norm(ffn_inp, model.layers[il].ffn_norm, NULL, LLM_NORM_RMS, il)`
- **調用點**: `cb(cur, "ffn_norm", il);`

### 數據特性：
- **數據類型**: 通常為 F32 (float)
- **維度**: `[n_embd]` (TinyLlama: 2048)
- **頻率**: 每層調用一次 (TinyLlama: 22層)
- **影響範圍**: 影響後續所有 FFN 計算和剩餘層

## 🧪 實驗設計方案

### Phase 1: 代碼修改
#### 1.1 擴展 fake quantization 參數
```cpp
// common/common.h 添加新參數
bool fake_quant_ffn_norm_enabled = false;     // 啟用FFN norm假量化
int fake_quant_target_layer = -1;             // 目標層號 (-1=所有層)
float fake_quant_ffn_scale = 1.0f;            // FFN norm量化比例
```

#### 1.2 命令行參數
```bash
--fake-quant-ffn-norm TYPE      # 啟用FFN norm假量化
--fake-quant-layer N           # 指定目標層號 (-1=所有層)
--fake-quant-ffn-scale FLOAT   # FFN norm量化比例 (0.0-1.0)
```

#### 1.3 核心實現位置
在 `src/llama-model.cpp` 的 `cb(cur, "ffn_norm", il)` 之後添加：
```cpp
// 在每個 cb(cur, "ffn_norm", il) 後添加
if (cparams.fake_quant_ffn_norm_enabled && 
    (cparams.fake_quant_target_layer == -1 || cparams.fake_quant_target_layer == il)) {
    
    // 獲取 tensor 數據
    float* data = (float*)cur->data;
    size_t n_elements = ggml_nelements(cur);
    
    // 應用假量化
    llama_fake_quantize_data(data, n_elements, cparams.fake_quant_type);
    
    // Debug 輸出
    if (cparams.fake_quant_compare) {
        LLAMA_LOG_INFO("Applied fake quantization to ffn_norm layer %d\n", il);
    }
}
```

### Phase 2: 實驗設計

#### 2.1 測試矩陣
| 實驗組 | FFN Norm量化 | Logits量化 | 目標層 | 量化類型 |
|--------|-------------|-----------|--------|---------|
| 基準組 | ❌ | ❌ | - | - |
| 實驗組1 | ✅ | ❌ | 所有層 | BF16 |
| 實驗組2 | ✅ | ❌ | 前5層 | BF16 |
| 實驗組3 | ✅ | ❌ | 後5層 | BF16 |
| 實驗組4 | ✅ | ✅ | 所有層 | BF16 |
| 實驗組5 | ✅ | ❌ | 所有層 | F16 |

#### 2.2 測試場景
```bash
# 基準測試
./build/bin/llama-cli -m model.gguf -p "測試提示" -n 50 --seed 42

# FFN norm BF16 假量化 (所有層)
./build/bin/llama-cli -m model.gguf --fake-quant-ffn-norm bf16 \
  -p "測試提示" -n 50 --seed 42

# FFN norm BF16 假量化 (指定層)
./build/bin/llama-cli -m model.gguf --fake-quant-ffn-norm bf16 \
  --fake-quant-layer 10 -p "測試提示" -n 50 --seed 42

# 組合測試 (FFN norm + Logits)
./build/bin/llama-cli -m model.gguf --fake-quant-ffn-norm bf16 \
  --fake-quant bf16 -p "測試提示" -n 50 --seed 42
```

### Phase 3: 測試指標

#### 3.1 數值精度指標
- **FFN輸出變化**: 比較FFN層輸出的數值差異
- **累積誤差**: 測量通過多層後的誤差累積
- **激活分佈**: 分析激活值分佈的變化

#### 3.2 生成質量指標
- **文本一致性**: 比較生成文本的差異
- **Token選擇**: 分析token選擇概率的變化
- **語義連貫性**: 評估語義理解的影響

#### 3.3 性能指標
- **計算開銷**: 測量假量化的性能影響
- **記憶體使用**: 監控記憶體使用變化

### Phase 4: 預期影響分析

#### 4.1 影響程度預測
```
FFN Norm 假量化影響 >> Logits 假量化影響

原因：
1. 早期影響：FFN norm在推理早期，影響後續所有計算
2. 累積效應：每層的精度損失會累積到最終輸出
3. 數值敏感性：Normalization對數值精度較敏感
```

#### 4.2 敏感性分析
| 層位置 | 預期影響 | 原因 |
|--------|----------|------|
| 前幾層 | 高影響 | 影響後續所有層的計算 |
| 中間層 | 中等影響 | 部分累積效應 |
| 後幾層 | 較低影響 | 接近輸出，累積效應有限 |

## 🔧 實現檢查清單

### 必要修改點：
- [ ] `common/common.h`: 添加FFN norm假量化參數
- [ ] `common/arg.cpp`: 添加命令行參數解析
- [ ] `include/llama.h`: 添加API參數
- [ ] `src/llama-cparams.h`: 添加內部參數
- [ ] `src/llama-context.cpp`: 參數傳遞
- [ ] `src/llama-model.cpp`: 在所有`cb(cur, "ffn_norm", il)`後添加假量化

### 測試點位置 (src/llama-model.cpp)：
```cpp
// 需要修改的位置 (部分範例)
Line 5083: cb(cur, "ffn_norm", il);           // LLAMA架構
Line 5097: cb(cur, "ffn_norm", il);           // LLAMA MoE
Line 5256: cb(cur, "ffn_norm", il);           // LLAMA 變種
Line 6197: cb(cur, "ffn_norm", il);           // BAICHUAN
Line 6294: cb(cur, "ffn_norm", il);           // BAICHUAN 變種
... (約50+個位置需要修改)
```

## 📊 預期實驗結果

### 假設 1: 顯著影響
- FFN norm假量化會比logits假量化產生更明顯的輸出差異
- 早期層的假量化影響 > 後期層的假量化影響

### 假設 2: 累積效應
- 多層FFN norm假量化會產生累積的精度損失
- BF16 vs F16會展現不同程度的影響

### 假設 3: 敏感性差異
- 不同類型的任務對FFN norm精度敏感性不同
- 數學計算類任務 > 文本生成類任務

## ⚠️ 實現注意事項

1. **性能影響**: FFN norm在每層都調用，需要優化實現
2. **Memory Safety**: 確保tensor數據訪問安全
3. **架構兼容性**: 需要適配所有支持的模型架構
4. **Debug支持**: 添加詳細的debug輸出便於分析

## 📈 成功標準

1. ✅ **功能正確性**: 假量化正確應用到FFN norm層
2. ✅ **明顯差異**: 能觀察到比logits假量化更明顯的影響
3. ✅ **層級效應**: 不同層的假量化展現不同影響程度
4. ✅ **性能可接受**: 假量化開銷 < 10%

此實驗將為理解Transformer模型中不同位置的精度敏感性提供重要數據，並為未來的量化策略優化提供指導。