# MXFP4 量化實現分析報告

## 執行摘要

在 Llama 3.2 1B 量化實驗中發現，使用 `MXFP4_MOE` 量化格式會回退到 `Q8_0`，產生與 Q8_0 完全相同的結果。本報告深入分析了 llama.cpp 中 MXFP4 量化的實現機制，找出了回退原因，並提供了完整的技術分析。

## 問題現象

### 觀察到的異常行為
- **期望**：MXFP4_MOE 產生 4-bit 精度的量化模型
- **實際**：生成與 Q8_0 完全相同的結果
- **證據**：
  - 檔案大小相同：1.22GB
  - Perplexity 完全相同：10.2884 ± 0.07116
  - 推理性能完全相同：364 t/s 文字生成速度
  - 張量類型顯示為 `q8_0` 而非 `mxfp4`

## 根本原因分析

### 1. MXFP4_MOE 的設計意圖

**MXFP4_MOE 專門為 MoE (Mixture of Experts) 架構設計**，不適用於標準 Transformer 模型。

**核心判斷邏輯** (`/workspace/llama.cpp/src/llama-quant.cpp:229-236`)：
```c
} else if (ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE) {
    // MoE   tensors -> MXFP4
    // other tensors -> Q8_0
    if (tensor->ne[2] > 1) {
        new_type = GGML_TYPE_MXFP4;
    } else {
        new_type = GGML_TYPE_Q8_0;
    }
}
```

### 2. 張量維度檢查機制

**關鍵條件**：`tensor->ne[2] > 1`

這個條件檢查張量的第三個維度是否大於1，用於識別 MoE 專家張量：

#### 標準 FFN 張量（2D）
```c
// 普通模型：{n_ff, n_embd}
layer.ffn_down = create_tensor(..., {n_ff, n_embd}, 0);
// ne[0] = n_ff, ne[1] = n_embd, ne[2] = 1 (默認)
```

#### MoE FFN 張量（3D）  
```c
// MoE 模型：{n_ff, n_embd, n_expert}
layer.ffn_down_exps = create_tensor(..., {n_ff, n_embd, n_expert}, 0);
// ne[0] = n_ff, ne[1] = n_embd, ne[2] = n_expert > 1
```

### 3. Llama 3.2 1B 的架構特性

**Llama 3.2 1B 不是 MoE 模型**：
- `n_expert = 0` (默認值，定義在 `/workspace/llama.cpp/src/llama-hparams.h:47`)
- 所有 FFN 層都是標準的 2D 張量
- 沒有專家混合機制

**結果**：所有張量的 `ne[2] <= 1`，觸發回退機制，強制使用 Q8_0。

## 技術實現細節

### 1. MXFP4 數據結構定義

**位置**：`/workspace/llama.cpp/ggml/src/ggml-common.h:190-195`
```c
#define QK_MXFP4 32
typedef struct {
    uint8_t e;              // E8M0 - 8位指數
    uint8_t qs[QK_MXFP4/2]; // 量化權重，每個 4-bit
} block_mxfp4;
```

**特點**：
- 每個 block 包含 32 個元素
- 共享 8-bit 指數 + 16 個 4-bit 元素
- 總共 17 bytes per block，平均 4.25 bits per weight

### 2. 量化類型決策流程

```
用戶指定 MXFP4_MOE
     ↓
設置 default_type = GGML_TYPE_MXFP4
     ↓
對每個張量調用 llama_tensor_get_type()
     ↓
檢查: ftype == LLAMA_FTYPE_MOSTLY_MXFP4_MOE?
     ↓ (Yes)
檢查: tensor->ne[2] > 1?
     ↓
 ┌────────┐    ┌────────┐
 │Yes     │    │No      │
 │GGML_   │    │GGML_   │
 │TYPE_   │    │TYPE_   │
 │MXFP4   │    │Q8_0    │
 └────────┘    └────────┘
    (MoE)      (Standard)
```

### 3. 量化過程驗證

**實際量化日誌分析**：
```
[   3/ 147] token_embd.weight - converting to q8_0
[   4/ 147] blk.0.attn_k.weight - converting to q8_0  
[   6/ 147] blk.0.attn_output.weight - converting to q8_0
[   9/ 147] blk.0.ffn_down.weight - converting to q8_0
```

**關鍵觀察**：
- 所有權重都顯示 "converting to q8_0"
- 沒有任何張量使用 MXFP4 格式
- 最終檔案類型標記為 "MXFP4 MoE" 但實際內容是 Q8_0

## 設計合理性評估

### 1. 為什麼這樣設計？

**MoE 模型的特殊需求**：
- MoE 模型有大量專家參數（通常 8-64 個專家）
- 專家權重經常處於"冷態"（不被激活）
- 需要更激進的壓縮來節省記憶體
- 4-bit MXFP4 格式適合專家權重的壓縮需求

**標準模型的考量**：
- 所有參數都經常被使用
- 需要保持較高精度以維持性能
- Q8_0 已經是很好的平衡點

### 2. 是否為錯誤？

**這不是錯誤，是有意的設計決策**：
- 代碼中有清晰的條件判斷邏輯
- 註釋明確標示："MoE tensors -> MXFP4, other tensors -> Q8_0"
- 這確保了量化工具不會錯誤地對標準模型應用不適當的量化

## 解決方案與建議

### 1. 對於 Llama 3.2 1B

**推薦做法**：
- 使用 `Q4_K_M` - 最佳的品質/性能/大小平衡
- 使用 `Q5_K_M` - 如果需要更高品質
- 使用 `Q8_0` - 如果需要最高品質
- **避免使用 `MXFP4_MOE`** - 它會回退到 Q8_0

### 2. 對於真正的 MoE 模型

**適用模型**：
- Mixtral 8x7B
- Mixtral 8x22B
- Switch Transformer 系列
- 其他 MoE 架構

**使用方法**：
```bash
./build/bin/llama-quantize moe_model.gguf moe_model_mxfp4.gguf MXFP4_MOE
```

### 3. 檢查模型是否為 MoE

**方法一**：檢查模型 metadata
```bash
./build/bin/llama-quantize model.gguf --help 2>/dev/null | grep expert
```

**方法二**：檢查層結構
- 尋找 `ffn_gate_exps`, `ffn_down_exps`, `ffn_up_exps` 等層名
- 檢查 `n_expert` 參數是否 > 0

## 量化格式選擇指南

### 標準 Transformer 模型（如 Llama 3.2 1B）
| 格式 | 使用場景 | 品質 | 大小 | 速度 |
|------|----------|------|------|------|
| Q4_K_M | 生產環境推薦 | 良好 | 763MB | 487 t/s |
| Q5_K_M | 高品質應用 | 優秀 | 862MB | 484 t/s |
| Q8_0 | 品質優先 | 最佳 | 1.22GB | 364 t/s |
| ❌ MXFP4_MOE | 不適用 | N/A | N/A | N/A |

### MoE 模型
| 格式 | 使用場景 | 說明 |
|------|----------|------|
| MXFP4_MOE | 專家壓縮 | 對專家使用 4-bit，其他用 Q8_0 |
| Q4_K_M | 平衡選擇 | 統一 4-bit 量化 |
| Q8_0 | 品質優先 | 統一 8-bit 量化 |

## 結論

MXFP4_MOE 量化格式的回退行為是**有意的設計特性**，而非錯誤。它確保了：

1. **正確性**：防止對不適當的模型應用錯誤的量化策略
2. **效能**：標準模型獲得適合的量化精度
3. **一致性**：維持預期的模型品質

**對於 Llama 3.2 1B 等標準模型，建議使用 Q4_K_M 或 Q5_K_M 格式以獲得最佳效果。**

---

**分析日期**：2025-08-07  
**分析環境**：llama.cpp commit c659b10e  
**測試模型**：Llama 3.2 1B Original