# TinyLlama FFN Norm Fake Quantization 技術確認分析

## 🎯 問題確認

**用戶問題**: 確認 TinyLlama 中 `blk.21.ffn_norm.weight` 是否會受到實驗設計中的 fake quantization 影響，並提供需要修改的具體函數。

## ✅ 關鍵發現：確認會受到影響

### 1. TinyLlama 架構確認
- **模型架構**: `LLM_ARCH_LLAMA` (標準 LLaMA 架構)
- **層數**: 22層 (0-21)，所以 `blk.21` 是最後一層
- **FFN Norm類型**: `LLM_NORM_RMS` (RMS Normalization)

### 2. 具體調用路徑分析

#### 2.1 FFN Norm 權重創建
**位置**: `src/llama-model.cpp:1898`
```cpp
layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
```
- TinyLlama的 `blk.21.ffn_norm.weight` 在此處創建
- 維度: `{2048}` (n_embd = 2048)

#### 2.2 前向傳播調用路徑
**位置**: `src/llama-model.cpp:5080-5083` (llm_build_llama函數)
```cpp
cur = build_norm(ffn_inp,
        model.layers[il].ffn_norm, NULL,
        LLM_NORM_RMS, il);
cb(cur, "ffn_norm", il);
```

#### 2.3 build_norm 函數實現
**位置**: `src/llama-graph.cpp:475-484`
```cpp
ggml_tensor * llm_graph_context::build_norm(
         ggml_tensor * cur,
         ggml_tensor * mw,
         ggml_tensor * mb,
       llm_norm_type   type,
                 int   il) const {
    switch (type) {
        case LLM_NORM_RMS:   cur = ggml_rms_norm(ctx0, cur, hparams.f_norm_rms_eps); break;
        // ...
    }
    
    if (mw) {
        cur = ggml_mul(ctx0, cur, mw);  // 這裡使用 ffn_norm weight
    }
    return cur;
}
```

### 3. Fake Quantization 應用點確認

#### 3.1 當前調用點
```cpp
// src/llama-model.cpp:5083
cb(cur, "ffn_norm", il);  // il=21 時，這裡會被調用
```

#### 3.2 需要添加的 Fake Quantization 代碼
**在 `cb(cur, "ffn_norm", il);` 之後立即添加:**
```cpp
// 在每個 cb(cur, "ffn_norm", il) 後添加
if (cparams.fake_quant_ffn_norm_enabled && 
    (cparams.fake_quant_target_layer == -1 || cparams.fake_quant_target_layer == il)) {
    
    // 確保 tensor 是 F32 類型且有數據
    if (cur->type == GGML_TYPE_F32 && cur->data != nullptr) {
        float* data = (float*)cur->data;
        size_t n_elements = ggml_nelements(cur);
        
        // 應用假量化
        llama_fake_quantize_data(data, n_elements, cparams.fake_quant_type);
        
        // Debug 輸出
        if (cparams.fake_quant_compare) {
            LLAMA_LOG_INFO("Applied fake quantization to ffn_norm layer %d\n", il);
        }
    }
}
```

## 🔧 需要修改的具體函數和位置

### 1. 參數結構體擴展
**文件**: `src/llama-cparams.h`
```cpp
struct llama_cparams {
    // 現有參數...
    
    // 新增 FFN norm fake quantization 參數
    bool fake_quant_ffn_norm_enabled = false;  // 啟用FFN norm假量化
    int  fake_quant_target_layer = -1;         // 目標層號 (-1=所有層)
    float fake_quant_ffn_scale = 1.0f;         // FFN norm量化比例
};
```

### 2. 命令行參數解析
**文件**: `common/arg.cpp`
```cpp
// 添加新的命令行參數
{"fake-quant-ffn-norm",  required_argument, 0, 0},
{"fake-quant-layer",     required_argument, 0, 0},
{"fake-quant-ffn-scale", required_argument, 0, 0},
```

### 3. API接口擴展
**文件**: `include/llama.h`
```cpp
struct llama_context_params {
    // 現有參數...
    bool  fake_quant_ffn_norm_enabled;
    int   fake_quant_target_layer;
    float fake_quant_ffn_scale;
};
```

### 4. 核心修改位置列表

#### 需要在所有以下位置添加 fake quantization 代碼:
```cpp
// LLAMA 架構 (TinyLlama 使用此架構)
src/llama-model.cpp:5083   // 非MoE版本
src/llama-model.cpp:5097   // MoE版本

// 其他架構（為了完整支持）
src/llama-model.cpp:5256   // LLAMA變種
src/llama-model.cpp:6197   // BAICHUAN
src/llama-model.cpp:6294   // BAICHUAN變種
// ... 約50+個其他架構位置
```

### 5. 關鍵修改模板
```cpp
// 在每個 cb(cur, "ffn_norm", il); 之後添加
if (cparams.fake_quant_ffn_norm_enabled && 
    (cparams.fake_quant_target_layer == -1 || cparams.fake_quant_target_layer == il)) {
    
    if (cur->type == GGML_TYPE_F32 && cur->data != nullptr) {
        float* data = (float*)cur->data;
        size_t n_elements = ggml_nelements(cur);
        llama_fake_quantize_data(data, n_elements, cparams.fake_quant_type);
        
        if (cparams.fake_quant_compare) {
            LLAMA_LOG_INFO("FFN norm fake quant applied: layer %d, elements %zu\n", il, n_elements);
        }
    }
}
```

## ✅ 確認結果

### 1. **會受到影響**: ✅ 確認
- TinyLlama 的 `blk.21.ffn_norm.weight` 會在 `llm_build_llama` 函數中被處理
- 該權重會通過 `build_norm` → `ggml_rms_norm` → `ggml_mul` 的路徑影響計算
- 在 `cb(cur, "ffn_norm", 21)` 調用後，`cur` tensor 包含了經過 ffn_norm 處理的結果

### 2. **數據流確認**: ✅ 正確
```
ffn_inp → build_norm(ffn_inp, ffn_norm_weight) → cur → cb(cur, "ffn_norm", 21) → [我們的fake quant]
```

### 3. **影響範圍**: ✅ 符合預期
- **直接影響**: FFN 層的輸入歸一化結果
- **累積影響**: 影響後續 FFN 層計算和最終輸出
- **敏感性**: 比 logits 假量化更早介入，影響更大

### 4. **技術可行性**: ✅ 完全可行
- 現有 `llama_fake_quantize_data` 函數可直接使用
- 數據類型為 F32，適合假量化處理
- 數據已在內存中，可以直接修改

## 🧪 測試驗證方法

### 測試命令示例:
```bash
# 基準測試 (無假量化)
./build/bin/llama-cli -m tinyllama.gguf -p "測試" -n 10 --seed 42

# 測試 layer 21 的 FFN norm BF16 假量化
./build/bin/llama-cli -m tinyllama.gguf --fake-quant-ffn-norm bf16 \
  --fake-quant-layer 21 -p "測試" -n 10 --seed 42

# 測試所有層的 FFN norm BF16 假量化  
./build/bin/llama-cli -m tinyllama.gguf --fake-quant-ffn-norm bf16 \
  -p "測試" -n 10 --seed 42
```

### 預期結果:
- 能看到 `"FFN norm fake quant applied: layer 21"` 的日志輸出
- 與基準相比會有明顯的輸出差異（比 logits 假量化影響更大）
- layer 21 單獨測試影響相對較小，全層測試影響顯著

## 📊 結論

**✅ 確認**: TinyLlama 的 `blk.21.ffn_norm.weight` **會被實驗設計真實影響**

1. **路徑確認**: 權重會在前向傳播中被使用，計算結果會被假量化處理
2. **位置確認**: `src/llama-model.cpp:5083` 是關鍵修改點
3. **影響確認**: 假量化會影響 FFN 層的歸一化輸出，進而影響整個推理結果
4. **實現確認**: 技術實現完全可行，修改點明確

**建議**: 可以開始實施，預期會觀察到比現有 logits 假量化更明顯的影響。