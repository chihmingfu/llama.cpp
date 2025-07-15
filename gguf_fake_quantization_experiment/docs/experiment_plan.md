# GGUF FFN Norm Fake Quantization 實驗計劃

## 實驗目標

本實驗旨在研究將GGUF模型中的FFN (Feed Forward Network) norm權重進行fake quantization的效果，具體目標包括：

1. **模擬BF16精度損失**：將F32精度的FFN norm權重轉換成BF16精度，但仍以F32格式存儲
2. **保持模型可用性**：確保fake quantized後的模型仍可正常載入和推理
3. **量化精度損失分析**：比較原始模型和fake quantized模型的數值差異
4. **模型性能評估**：分析perplexity變化，評估量化對模型質量的影響
5. **實驗可重現性**：建立完整的實驗流程和記錄，便於後續研究

## 技術背景

### BF16 (bfloat16) 格式
- **位元配置**：1位符號 + 8位指數 + 7位尾數
- **範圍**：與FP32相同的指數範圍，但精度較低
- **轉換方式**：從FP32的32位中保留高16位，捨棄低16位

### Fake Quantization 概念
- **定義**：模擬量化過程中的精度損失，但保持原始數據格式
- **用途**：分析量化對模型的影響，不需要實際部署量化格式
- **優勢**：可直接使用現有FP32推理框架進行測試

## 實驗設計

### 階段一：環境準備和工具開發

#### 1.1 目錄結構
```
gguf_fake_quantization_experiment/
├── scripts/           # 實驗腳本
├── data/             # 測試數據
├── results/          # 實驗結果
├── docs/             # 文檔和計劃
└── logs/             # 執行日誌
```

#### 1.2 開發工具
- **fake_quantize_gguf.py**：主要的fake quantization工具
- **compare_models.py**：模型差異比較工具
- **evaluate_model.py**：模型性能評估工具
- **experiment_runner.py**：自動化實驗執行器

#### 1.3 測試模型選擇
- **主要測試模型**：tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
- **原因**：模型較小，便於快速測試和驗證

### 階段二：Fake Quantization 實現

#### 2.1 核心演算法
```python
def fake_quantize_to_bf16(tensor_f32):
    """
    將F32 tensor進行BF16 fake quantization
    
    步驟：
    1. 將F32數據轉換為uint32位表示
    2. 右移16位，保留高16位（符號+指數+部分尾數）
    3. 左移16位，回到F32格式（低16位填0）
    4. 轉換回F32格式
    """
    # 保留BF16精度的F32數據
    f32_bits = tensor_f32.view(np.uint32)
    bf16_bits = (f32_bits >> 16) << 16
    return bf16_bits.view(np.float32)
```

#### 2.2 目標權重識別
- **目標層**：所有包含 `.ffn_norm.weight` 的tensor
- **選擇原因**：FFN norm權重對模型性能影響相對較小，適合量化實驗
- **範圍控制**：可指定特定層範圍進行測試

#### 2.3 實現步驟
1. 讀取原始GGUF文件
2. 識別所有FFN norm權重
3. 對目標權重進行fake BF16 quantization
4. 保持其他權重和metadata不變
5. 寫入新的GGUF文件

### 階段三：實驗執行

#### 3.1 基線測試
```bash
# 測試原始模型
./build/bin/llama-perplexity -m models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf \
    -f wikitext-2-raw/wiki.test.small.raw --threads 8 --ctx-size 512 --batch-size 8
```

#### 3.2 實驗執行順序
**優先執行：測試1 - 單層fake quantization (layer 0)**
- 根據測試1結果決定後續實驗步驟
- 如果測試1成功，再考慮測試2-4的執行

**後續測試（依測試1結果決定）**：
- 測試2：前3層fake quantization (layers 0-2)
- 測試3：前10層fake quantization (layers 0-9)  
- 測試4：全部層fake quantization (all layers)

#### 3.3 測試1詳細內容
1. **數值差異分析**：計算layer 0 FFN norm權重的fake quantization前後差異
2. **模型載入測試**：確認修改後的模型可正常載入到llama.cpp
3. **對話功能測試**：使用llama-cli測試token生成和對話能力
4. **Perplexity評估**：使用標準數據集評估模型質量變化
5. **綜合評估**：決定是否繼續後續測試

### 階段四：結果分析

#### 4.1 數值分析指標
- **絕對差異**：`|original - fake_quantized|`
- **相對差異**：`|original - fake_quantized| / (|original| + ε)`
- **最大差異**：每個tensor的最大絕對差異
- **平均差異**：每個tensor的平均絕對差異
- **分佈變化**：權重分佈的統計特性變化

#### 4.2 性能評估指標
- **Perplexity變化**：量化前後的perplexity差異
- **推理速度**：是否有性能影響（理論上應該沒有）
- **輸出一致性**：相同輸入的輸出相似度

#### 4.3 結果記錄
- **數值結果表**：各層量化效果的詳細數值
- **性能比較表**：perplexity和推理性能對比
- **可視化圖表**：權重分佈變化和差異分佈

## 實驗流程

### Step 1: 準備工作
```bash
# 創建實驗環境
cd gguf_fake_quantization_experiment

# 複製測試模型到data目錄
cp ../models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf data/original_model.gguf

# 準備測試數據
cp ../wikitext-2-raw/wiki.test.small.raw data/
```

### Step 2: 開發和測試工具（專注測試1）
```bash
# 開發fake quantization工具
python scripts/fake_quantize_gguf.py data/original_model.gguf data/fake_quant_layer0.gguf --layers "0"

# 測試模型載入
./build/bin/llama-cli -m data/fake_quant_layer0.gguf --prompt "Hello" -n 10

# 對話功能測試
./build/bin/llama-cli -m data/fake_quant_layer0.gguf -i --prompt "你好，請介紹一下自己。" -n 50

# 執行perplexity測試
./build/bin/llama-perplexity -m data/fake_quant_layer0.gguf -f data/wiki.test.small.raw --threads 8 --ctx-size 512 --batch-size 8

# 數值差異分析
python scripts/compare_models.py data/original_model.gguf data/fake_quant_layer0.gguf --focus-layer 0
```

### Step 3: 測試1結果評估和決策
```bash
# 基於測試1結果生成報告
python scripts/generate_test1_report.py results/layer0/ docs/test1_results.md

# 根據測試1結果決定是否進行後續測試
# （此步驟將根據用戶對測試1結果的評估來決定）
```

### Step 4: 後續實驗（如測試1成功）
```bash
# 僅在測試1成功且用戶決定繼續時執行
# 測試不同範圍的量化
for range in "0,2" "0,9" "all"; do
    echo "Testing fake quantization for layers: $range"
    python scripts/fake_quantize_gguf.py data/original_model.gguf data/fake_quant_${range}.gguf --layers "$range"
    python scripts/evaluate_model.py data/fake_quant_${range}.gguf --baseline data/original_model.gguf
done
```

### Step 4: 結果分析和報告
```bash
# 生成比較報告
python scripts/generate_report.py results/ docs/experiment_results.md

# 創建可視化圖表
python scripts/visualize_results.py results/ docs/charts/
```

## 預期結果

### 數值層面
- **BF16精度損失範圍**：預期最大相對差異在 10^-3 到 10^-2 之間
- **權重分佈**：整體分佈形狀保持相似，但細節精度降低
- **層數影響**：量化層數越多，累積誤差越大

### 性能層面
- **Perplexity變化**：預期輕微增加（模型質量輕微下降）
- **推理可用性**：模型應保持可用，輸出合理
- **量化敏感性**：不同層對量化的敏感度可能不同

### 實際應用價值
- **量化可行性評估**：為實際BF16量化提供參考
- **敏感層識別**：找出對量化最敏感的層
- **性能預估**：為實際量化部署提供性能預期

## 風險評估和緩解策略

### 潛在風險
1. **模型損壞**：fake quantization可能導致模型無法正常載入
2. **數值溢出**：BF16轉換過程中可能出現數值問題
3. **性能大幅下降**：量化可能嚴重影響模型性能

### 緩解策略
1. **逐步測試**：從單層開始，逐步增加量化範圍
2. **備份原始模型**：確保可以回退到原始狀態
3. **數值檢查**：在每個步驟添加數值有效性檢查
4. **early stopping**：如果發現嚴重問題立即停止實驗

## 實驗記錄要求

### 命令記錄
- 所有執行的bash命令
- 命令執行的時間戳和結果
- 錯誤訊息和處理方式

### 數據記錄
- 每次實驗的輸入參數
- 數值分析結果
- 性能測試結果
- 異常情況記錄

### 互動記錄
- 實驗過程中的問題和解決方案
- 計劃調整的原因和過程
- 技術決策的依據和考量

## 後續研究方向

### 量化格式擴展
- INT8 fake quantization
- 其他低精度格式的測試
- 混合精度量化策略

### 量化層範圍研究
- 不同類型權重的量化敏感性
- 最佳量化策略探索
- 層重要性分析

### 實際部署優化
- 基於fake quantization結果的實際量化實現
- 量化感知訓練的可行性評估
- 推理框架的適配和優化

---

## 實驗狀態追蹤

| 階段 | 狀態 | 開始時間 | 完成時間 | 備註 |
|------|------|----------|----------|------|
| 環境準備 | 計劃中 | - | - | - |
| 工具開發 | 計劃中 | - | - | - |
| 基線測試 | 計劃中 | - | - | - |
| 漸進測試 | 計劃中 | - | - | - |
| 結果分析 | 計劃中 | - | - | - |

**實驗負責人**: Jimmy Fu  
**計劃制定日期**: 2025-07-15  
**預計完成日期**: TBD  
**最後更新**: 2025-07-15  