# GGUF FFN Norm Fake Quantization 實驗

一個用於研究GGUF模型FFN norm權重BF16 fake quantization效果的實驗項目。

## 🎯 實驗目的

- 開發BF16 fake quantization工具
- 分析現代GGUF模型的權重精度分布
- 評估量化對模型性能的影響
- 建立可重現的量化實驗流程

## 🔍 主要發現

### 關鍵洞察
✅ **現代GGUF模型的norm權重已經被優化為BF16精度**  
✅ **Fake quantization工具實現正確，方法可靠**  
✅ **建立了完整的權重精度分析流程**  

### 實驗結果
- **TinyLlama 1.1B**: 所有F32 norm權重都是BF16精度，fake quantization無效果
- **Llama-3.2 1B**: 驗證了工具正確性，rope_freqs.weight產生可測量的量化效果
- **Perplexity測試**: 在BF16精度權重上無變化（預期結果）

## 🛠 工具與方法

### 核心工具
- **fake_quantize_gguf.py**: 主要的BF16 fake quantization工具
- **精度分析腳本**: 檢查權重的低16位分布
- **模型比較工具**: 對比量化前後的模型效果

### 量化算法
```python
def fake_quantize_to_bf16(tensor_f32):
    """BF16 fake quantization: 保留高16位，清零低16位"""
    f32_bits = tensor_f32.astype(np.float32).view(np.uint32)
    bf16_bits = (f32_bits >> 16) << 16
    return bf16_bits.view(np.float32)
```

## 📁 項目結構

```
gguf_fake_quantization_experiment/
├── scripts/
│   └── fake_quantize_gguf.py          # 主要量化工具
├── data/                              # 測試模型和數據
├── results/                           # 實驗結果和分析
├── docs/                              # 實驗計劃和文檔
├── logs/                              # 命令記錄和日誌
├── FFN_NORM_FAKE_QUANTIZATION_FINAL_REPORT.md  # 最終報告
└── README.md                          # 本文件
```

## 🚀 快速開始

### 環境準備
```bash
# 確保已安裝llama.cpp和gguf-py
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
make

# 安裝Python依賴
pip install numpy
```

### 運行實驗
```bash
# 進入實驗目錄
cd gguf_fake_quantization_experiment

# 運行fake quantization（使用默認參數）
python scripts/fake_quantize_gguf.py --verbose

# 檢查權重精度分布
python check_f32_norms.py
```

### 自定義測試
```bash
# 指定輸入和輸出模型
python scripts/fake_quantize_gguf.py input.gguf output.gguf --layers "0,1,2" --verbose

# 測試不同層
python scripts/fake_quantize_gguf.py input.gguf output.gguf --layers "5,10,15" --verbose
```

## 📊 實驗結果示例

### TinyLlama模型測試
```
Layer 0 FFN norm權重:
- 數值差異: 0.00% (已是BF16精度)
- Perplexity: 15.9228 (無變化)
- 對話質量: 完全一致
```

### Llama-3.2模型驗證
```
rope_freqs.weight:
- 最大絕對差異: 4.17e-02
- 平均絕對差異: 1.74e-03
- 最大相對差異: 0.43%
✅ 證明工具正確性
```

## 📖 詳細文檔

- [最終實驗報告](FFN_NORM_FAKE_QUANTIZATION_FINAL_REPORT.md) - 完整的實驗結果和分析
- [命令歷史記錄](logs/command_history.md) - 所有執行命令的詳細記錄
- [實驗計劃](docs/experiment_plan.md) - 原始實驗設計
- [精度分析報告](results/layer0/precision_analysis_report.md) - 權重精度分布分析

## 🔬 技術細節

### BF16 Fake Quantization原理
1. **讀取F32權重**: 載入32位浮點數權重
2. **位操作轉換**: 保留高16位（符號+指數+高7位尾數），清零低16位
3. **保持F32格式**: 轉換後仍以F32格式儲存，但精度等同BF16
4. **效果分析**: 計算量化前後的數值差異

### 權重精度檢測
```python
# 檢查權重是否為BF16精度
bits = tensor.view(np.uint32)
low_16_bits = bits & 0xFFFF
bf16_ratio = np.count_nonzero(low_16_bits == 0) / tensor.size
```

## 🎯 研究價值

### 技術貢獻
- **量化工具**: 提供通用的fake quantization實現
- **分析方法**: 建立權重精度分析標準流程
- **模型洞察**: 揭示現代GGUF模型的精度優化策略

### 實際應用
- **模型理解**: 幫助理解量化模型的內在特性
- **研究基礎**: 為進一步的量化研究提供工具和方法
- **性能分析**: 評估不同精度對模型性能的影響

## 🔮 後續方向

### 建議的擴展實驗
- [ ] 測試更多模型架構和大小
- [ ] 實現INT8、4-bit等其他量化方法
- [ ] 系統性分析不同層的量化敏感性
- [ ] 開發自動化的最優量化策略

### 工具改進
- [ ] 支援更多權重類型的量化
- [ ] 添加量化效果的自動評估
- [ ] 開發GUI界面
- [ ] 優化大模型的處理性能

## 📄 許可證

本項目採用與llama.cpp相同的許可證。

## 🤝 貢獻

歡迎提交Issue和Pull Request！

## 📧 聯繫

如有問題或建議，請通過GitHub Issues聯繫。

---

**實驗狀態**: ✅ 完成  
**最後更新**: 2025-07-15  
**技術驗證**: ✅ 工具正確，方法可靠