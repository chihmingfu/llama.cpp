# 測試1 詳細計劃：Layer 0 FFN Norm Fake Quantization

## 測試概述

**目標**: 對tinyllama模型的第0層FFN norm權重進行BF16 fake quantization，全面評估其對模型功能的影響。

**重要性**: 測試1的結果將決定是否繼續進行後續的多層量化實驗。

## 測試1檢查項目

### 1. 數值差異分析
- **目標權重**: `blk.0.ffn_norm.weight`
- **分析指標**:
  - 最大絕對差異: `max(|original - fake_quantized|)`
  - 平均絕對差異: `mean(|original - fake_quantized|)`
  - 最大相對差異: `max(|original - fake_quantized| / (|original| + ε))`
  - 平均相對差異: `mean(|original - fake_quantized| / (|original| + ε))`
  - 權重分佈變化

### 2. 模型載入測試
```bash
# 基本載入測試
./build/bin/llama-cli -m data/fake_quant_layer0.gguf --help

# 快速推理測試
./build/bin/llama-cli -m data/fake_quant_layer0.gguf --prompt "Hello" -n 5
```

### 3. 對話功能測試（重點新增）
```bash
# 英文對話測試
./build/bin/llama-cli -m data/fake_quant_layer0.gguf \
    --prompt "Hello, can you introduce yourself?" -n 50 --temp 0.7

# 中文對話測試  
./build/bin/llama-cli -m data/fake_quant_layer0.gguf \
    --prompt "你好，請介紹一下自己。" -n 50 --temp 0.7

# 簡單推理測試
./build/bin/llama-cli -m data/fake_quant_layer0.gguf \
    --prompt "2+2等於多少？" -n 20 --temp 0.1

# 對話連續性測試（交互模式）
./build/bin/llama-cli -m data/fake_quant_layer0.gguf -i \
    --prompt "讓我們聊聊天吧。" -n 30
```

### 4. Perplexity評估
```bash
# 原始模型基線
./build/bin/llama-perplexity -m data/original_model.gguf \
    -f data/wiki.test.small.raw --threads 8 --ctx-size 512 --batch-size 8

# Fake quantized模型
./build/bin/llama-perplexity -m data/fake_quant_layer0.gguf \
    -f data/wiki.test.small.raw --threads 8 --ctx-size 512 --batch-size 8
```

## 成功標準

### 必須通過的檢查
1. **模型載入**: 無錯誤載入
2. **基本推理**: 能生成合理的token
3. **數值穩定**: 沒有NaN或Inf值
4. **格式正確**: 輸出格式與原始模型一致

### 質量評估標準
1. **數值差異**: 
   - 平均相對差異 < 1%
   - 最大相對差異 < 10%
2. **對話質量**:
   - 能回應簡單問題
   - 生成語法正確的句子
   - 保持基本邏輯一致性
3. **Perplexity變化**:
   - 增加幅度 < 5%（可接受範圍）
   - 增加幅度 < 1%（理想範圍）

## 預期結果分析

### 樂觀情況
- 數值差異在可接受範圍內
- 對話功能正常，質量輕微下降
- Perplexity增加 < 1%
- **決策**: 繼續測試2-4

### 中等情況  
- 數值差異較大但可接受
- 對話功能正常，質量明顯但不嚴重下降
- Perplexity增加 1-5%
- **決策**: 謹慎繼續或調整方法

### 悲觀情況
- 數值差異過大
- 對話功能異常或質量嚴重下降
- Perplexity增加 > 5%
- **決策**: 停止實驗，檢討方法

## 實驗執行流程

### Phase 1: 準備工作
1. 複製原始模型到實驗目錄
2. 準備測試數據集
3. 建立結果記錄目錄

### Phase 2: Fake Quantization
1. 開發並測試fake quantization工具
2. 對layer 0執行fake quantization
3. 驗證新模型文件的完整性

### Phase 3: 評估測試
1. 執行數值差異分析
2. 進行對話功能測試
3. 執行perplexity評估
4. 記錄所有結果

### Phase 4: 結果分析和決策
1. 整理所有測試結果
2. 對照成功標準評估
3. 生成測試1報告
4. 決定後續實驗方向

## 風險管控

### 可能的問題
1. **工具錯誤**: fake quantization實現有bug
2. **數值問題**: 轉換過程產生異常值
3. **模型損壞**: GGUF文件結構被破壞
4. **性能劣化**: 量化影響超出預期

### 應對策略
1. **逐步驗證**: 每個步驟都檢查中間結果
2. **備份保護**: 保留原始模型文件
3. **早期檢測**: 在數值分析階段發現問題立即停止
4. **詳細記錄**: 記錄所有異常情況便於調試

## 輸出文件

### 結果文件
- `results/layer0/numerical_analysis.json` - 數值差異分析結果
- `results/layer0/conversation_tests.txt` - 對話測試輸出
- `results/layer0/perplexity_comparison.txt` - Perplexity比較結果
- `results/layer0/test1_summary.md` - 測試1總結報告

### 模型文件
- `data/original_model.gguf` - 原始模型（備份）
- `data/fake_quant_layer0.gguf` - Layer 0 fake quantized模型

---

**執行時間預估**: 1-2小時  
**關鍵決策點**: 測試完成後的go/no-go決策  
**後續行動**: 根據結果決定是否進行測試2-4  