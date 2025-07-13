# Fake Quantization with BF16 Support in llama.cpp

## 📋 研究總結與實現計劃

### 🎯 專案目標
實現 fake quantization 機制，使用 BF16 (Brain Float 16) 取代 FP16 或 FP32，以測試量化對模型精度的影響，同時保持計算效率。

---

## 🔍 現狀分析

### ✅ BF16 支持現狀
經過深入代碼分析，發現 **GGML 已經具備完整的 BF16 支援**：

#### 數據類型定義
```cpp
// ggml/include/ggml.h:389
enum ggml_type {
    GGML_TYPE_F32     = 0,
    GGML_TYPE_F16     = 1,
    // ... 其他量化類型
    GGML_TYPE_BF16    = 30,  // ✅ 已定義
    GGML_TYPE_COUNT   = 39,
};

// ggml/include/ggml.h:432
enum ggml_ftype {
    GGML_FTYPE_MOSTLY_BF16 = 24, // ✅ 檔案格式支援
};
```

#### 轉換函數
```cpp
// ggml/include/ggml.h:346-351
typedef struct { uint16_t bits; } ggml_bf16_t;
GGML_API ggml_bf16_t ggml_fp32_to_bf16(float);
GGML_API float       ggml_bf16_to_fp32(ggml_bf16_t);
GGML_API void        ggml_bf16_to_fp32_row(const ggml_bf16_t *, float *, int64_t);
GGML_API void        ggml_fp32_to_bf16_row(const float *, ggml_bf16_t *, int64_t);
```

#### 硬體加速支援
```cpp
// CMakeLists.txt 中的選項
option(GGML_AVX512_BF16 "ggml: enable AVX512-BF16" OFF)
option(GGML_AMX_BF16    "ggml: enable AMX-BF16"    OFF)
```

#### 向量化操作
```cpp
// ggml/src/ggml-cpu/vec.h:43
void ggml_vec_dot_bf16(int n, float * GGML_RESTRICT s, 
                       size_t bs, ggml_bf16_t * GGML_RESTRICT x, 
                       size_t bx, ggml_bf16_t * GGML_RESTRICT y, 
                       size_t by, int nrc);
```

### 🏗️ 現有量化架構
GGML 支援多種量化格式：
- **浮點數類型**: F32, F16, BF16, F64
- **整數量化**: Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
- **K-量化**: Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
- **IQ 系列**: IQ1_S, IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ4_NL, IQ4_XS
- **新型量化**: TQ1_0, TQ2_0

---

## 💡 BF16 vs FP16 比較

| 特性 | FP32 | FP16 | BF16 |
|------|------|------|------|
| **總位數** | 32 | 16 | 16 |
| **符號位** | 1 | 1 | 1 |
| **指數位** | 8 | 5 | 8 |
| **尾數位** | 23 | 10 | 7 |
| **數值範圍** | ±3.4×10³⁸ | ±6.5×10⁴ | ±3.4×10³⁸ |
| **精度** | ~7位十進位 | ~3位十進位 | ~2位十進位 |

### 🎯 BF16 優勢
1. **相同指數範圍**: 與 FP32 相同的 8-bit 指數，避免溢出問題
2. **更好的數值穩定性**: 較少的梯度消失/爆炸問題
3. **硬體支援**: Intel、AMD、NVIDIA 都有專門的 BF16 指令集
4. **記憶體效率**: 比 FP32 節省 50% 記憶體

---

## 🚀 Fake Quantization 實現方案

### 核心概念
**Fake Quantization** 是指在推理過程中模擬量化的精度損失，但不實際改變模型儲存格式：

1. **模型儲存**: 保持原始精度 (FP32/FP16)
2. **推理計算**: 動態轉換為目標精度 (BF16)
3. **精度損失**: 真實反映量化後的模型表現
4. **靈活性**: 可隨時切換不同精度模式

### 🏗️ 架構設計

#### 1. 參數配置
```cpp
// common/common.h - 新增參數
struct llama_context_params {
    // ... 現有參數
    bool fake_quant_enabled;           // 啟用 fake quantization
    enum ggml_type fake_quant_type;    // 目標量化類型 (如 GGML_TYPE_BF16)
    float fake_quant_scale;            // 量化比例 (0.0-1.0, 1.0=全部量化)
};
```

#### 2. 動態量化層
```cpp
// 新檔案: src/llama-fake-quant.h
#pragma once

#include "ggml.h"

// Fake quantization 函數
void llama_fake_quantize_tensor(
    struct ggml_tensor * tensor,
    enum ggml_type target_type,
    float scale_factor
);

// 批次處理
void llama_fake_quantize_model_tensors(
    struct llama_model * model,
    const struct llama_fake_quant_params * params
);
```

#### 3. 推理管道整合
```cpp
// src/llama-fake-quant.cpp
static void apply_fake_quantization_forward(
    const struct ggml_tensor * src,
    struct ggml_tensor * dst,
    enum ggml_type fake_type
) {
    switch (fake_type) {
        case GGML_TYPE_BF16: {
            // 動態轉換: FP32 -> BF16 -> FP32
            const float * src_data = (const float *)src->data;
            float * dst_data = (float *)dst->data;
            
            for (int64_t i = 0; i < ggml_nelements(src); i++) {
                // 模擬 BF16 精度損失
                ggml_bf16_t bf16_val = ggml_compute_fp32_to_bf16(src_data[i]);
                dst_data[i] = ggml_compute_bf16_to_fp32(bf16_val);
            }
            break;
        }
        // 支援其他量化類型...
    }
}
```

### 📁 檔案結構
```
src/
├── llama-fake-quant.h          # Fake quantization API
├── llama-fake-quant.cpp        # 核心實現
├── llama-context.cpp           # 整合到推理管道
└── llama-model.cpp            # 模型載入時的支援

tools/
├── main/main.cpp              # 命令行參數支援
└── server/server.cpp          # 伺服器 API 支援

common/
└── common.cpp                 # 參數解析
```

---

## 🛠️ 實施計劃

### Phase 1: 基礎實現 (1-2 週)
- [ ] 添加 fake quantization 命令行參數
- [ ] 實現基本的 BF16 fake quantization
- [ ] 整合到主要推理循環
- [ ] 基本功能測試

#### 命令行介面
```bash
# 啟用 BF16 fake quantization
./llama-cli -m model.gguf --fake-quant bf16 -p "Hello world"

# 部分量化 (50% 的層使用 BF16)
./llama-cli -m model.gguf --fake-quant bf16 --fake-quant-scale 0.5

# 對比模式 (同時輸出原始和量化結果)
./llama-cli -m model.gguf --fake-quant bf16 --compare-precision
```

### Phase 2: 優化與擴展 (2-3 週)
- [ ] 向量化 fake quantization 操作
- [ ] 支援混合精度 (不同層使用不同精度)
- [ ] 添加其他量化類型支援 (Q4_0, Q8_0 等)
- [ ] 性能優化

#### 混合精度配置
```cpp
// 配置檔案: fake_quant_config.json
{
    "layers": {
        "attention": "bf16",
        "feed_forward": "q4_0", 
        "embedding": "fp16",
        "output": "fp32"
    },
    "scale_factor": 1.0
}
```

### Phase 3: 評估與測試 (1-2 週)
- [ ] 精度損失分析工具
- [ ] 性能基準測試
- [ ] 記憶體使用分析
- [ ] 與真實量化模型對比

#### 評估工具
```bash
# 精度分析
./llama-bench --model model.gguf --fake-quant bf16 --precision-analysis

# 性能測試
./llama-bench --model model.gguf --fake-quant bf16 --performance-test

# 記憶體分析
./llama-cli --model model.gguf --fake-quant bf16 --memory-profile
```

---

## 📊 預期效果

### 精度影響
- **BF16**: 預期精度損失 < 5%
- **記憶體節省**: 理論上 0% (fake quantization 不改變儲存)
- **計算效率**: 在支援 BF16 的硬體上提升 10-30%

### 測試指標
1. **Perplexity 測試**: 使用 WikiText-2 dataset
2. **推理速度**: tokens/second 比較
3. **記憶體使用**: 峰值記憶體分析
4. **精度保持**: 與原模型的輸出差異

---

## 🔧 技術細節

### BF16 轉換實現
```cpp
// 已存在於 ggml-impl.h
static inline ggml_bf16_t ggml_compute_fp32_to_bf16(float s) {
    ggml_bf16_t h;
    union { float f; uint32_t i; } u;
    u.f = s;
    if ((u.i & 0x7fffffff) > 0x7f800000) { /* nan */
        h.bits = (u.i >> 16) | 64; /* force to quiet */
        return h;
    }
    h.bits = (u.i + (0x7fff + ((u.i >> 16) & 1))) >> 16;
    return h;
}

static inline float ggml_compute_bf16_to_fp32(ggml_bf16_t h) {
    union { float f; uint32_t i; } u;
    u.i = (uint32_t)h.bits << 16;
    return u.f;
}
```

### 向量化優化
```cpp
// 使用 AVX512BF16 指令集
#if defined(__AVX512BF16__)
void fake_quant_fp32_to_bf16_avx512(const float* src, float* dst, int n) {
    for (int i = 0; i + 16 <= n; i += 16) {
        __m512 fp32_vec = _mm512_loadu_ps(src + i);
        __m256i bf16_vec = _mm512_cvtneps_pbh(fp32_vec);
        __m512 result = _mm512_cvtpbh_ps(bf16_vec);
        _mm512_storeu_ps(dst + i, result);
    }
}
#endif
```

---

## 🚧 已知限制與挑戰

### 技術挑戰
1. **性能開銷**: 動態轉換可能影響推理速度
2. **記憶體拷貝**: 需要額外的緩衝區進行轉換
3. **精度控制**: 如何精確控制量化的應用範圍

### 解決方案
1. **惰性量化**: 只在必要時進行轉換
2. **In-place 操作**: 減少記憶體拷貝
3. **層級控制**: 細粒度的量化控制

---

## 📚 相關資源

### 程式碼位置
- **GGML BF16 實現**: `ggml/src/ggml-impl.h:436-470`
- **數據類型定義**: `ggml/include/ggml.h:358-398`
- **轉換函數**: `ggml/src/ggml.c:420-470`
- **向量操作**: `ggml/src/ggml-cpu/vec.h`

### 參考文獻
1. [Brain Floating Point Format](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format)
2. [Intel AVX512-BF16 Documentation](https://software.intel.com/content/www/us/en/develop/articles/intel-avx-512-bf16-instructions.html)
3. [GGML Documentation](https://github.com/ggml-org/ggml)

---

## ✅ 總結

這個研究發現 **GGML 已經具備完整的 BF16 基礎設施**，這大大簡化了實現 fake quantization 的複雜度。主要工作將集中在：

1. **應用層整合**: 將現有的 BF16 支援整合到推理管道
2. **用戶介面**: 提供簡單易用的命令行和 API 介面  
3. **性能優化**: 利用現有的向量化和硬體加速功能
4. **測試驗證**: 確保精度和性能符合預期

相比從零開始實現 BF16 支援，這個方案更加可行且風險較低。