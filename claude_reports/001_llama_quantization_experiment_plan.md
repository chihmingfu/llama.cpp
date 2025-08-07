# Llama 3.2 1B 量化實驗計畫

## 實驗目標
探索使用 llama.cpp 工具鏈從原始模型到量化模型的完整流程，並比較不同量化方法的效果。

## 實驗步驟

### 1. 下載原始模型

#### 1.1 環境準備
```bash
# 安裝 HuggingFace Hub
pip install huggingface-hub

# 建立模型目錄
mkdir -p models
```

#### 1.2 HuggingFace 登入和授權
```bash
# 設定 HuggingFace Token（替換為你的實際 token）
export HF_TOKEN="your_hf_token_here"

# 登入（使用新版指令）
huggingface-cli login --token "your_hf_token_here"
# 或使用新版本的指令
hf auth login --token "your_hf_token_here"
```

**重要：**
1. 需要先到 HuggingFace 上申請存取 meta-llama/Llama-3.2-1B 的權限
2. 接受 Meta 的授權條款
3. 在 HuggingFace 設定頁面生成 access token

#### 1.3 下載模型
```bash
# ✅ 實際測試成功的指令（使用新版 hf 命令）
hf download meta-llama/Llama-3.2-1B \
  --include "*.safetensors" "*.json" "tokenizer.model" \
  --local-dir ./models/llama-3.2-1b-original

# ❌ 已棄用的舊指令（會有警告）
# huggingface-cli download meta-llama/Llama-3.2-1B \
#   --include "*.safetensors" "*.json" "tokenizer.model" \
#   --local-dir ./models/llama-3.2-1b-original
```

#### 1.4 驗證下載
```bash
# 檢查下載的檔案
ls -la ./models/llama-3.2-1b-original/

# 檢查模型檔案大小（應為約 2.4GB）
ls -lh ./models/llama-3.2-1b-original/model.safetensors
```

**實際下載結果：**
- `model.safetensors`: 2.4GB（主要模型檔案）
- `config.json`: 模型配置
- `tokenizer.json`: Tokenizer 配置
- `generation_config.json`: 生成參數配置
- 其他支援檔案

**可能遇到的問題及解決方案：**

1. **401 Unauthorized 錯誤**：
   - 確認 HF token 設定正確
   - 確認已申請並獲得 Llama 3.2 存取權限
   - 重新登入 HuggingFace

2. **Gated Repository 錯誤**：
   - 必須在 HuggingFace 網站上申請存取權限
   - 接受 Meta 的授權條款

3. **指令過時警告**：
   - 使用新版 `hf download` 而非 `huggingface-cli download`

### 2. 轉換為 GGUF 格式
```bash
# 安裝必要的 Python 套件
pip install -r requirements-convert_hf_to_gguf.txt

# 轉換為 GGUF F16 格式（保持原始精度）
python convert_hf_to_gguf.py ./models/llama-3.2-1b-original \
  --outfile ./models/llama-3.2-1b-f16.gguf \
  --outtype f16
```

### 3. 量化實驗

#### 3.1 常用量化格式
```bash
# Q4_0 - 4-bit 量化（最小但品質較低）
./build/bin/llama-quantize ./models/llama-3.2-1b-f16.gguf \
  ./models/llama-3.2-1b-q4_0.gguf Q4_0

# Q4_K_M - 4-bit 量化（K-quants，平衡版）
./build/bin/llama-quantize ./models/llama-3.2-1b-f16.gguf \
  ./models/llama-3.2-1b-q4_k_m.gguf Q4_K_M

# Q5_K_M - 5-bit 量化（品質較好）
./build/bin/llama-quantize ./models/llama-3.2-1b-f16.gguf \
  ./models/llama-3.2-1b-q5_k_m.gguf Q5_K_M

# Q8_0 - 8-bit 量化（高品質）
./build/bin/llama-quantize ./models/llama-3.2-1b-f16.gguf \
  ./models/llama-3.2-1b-q8_0.gguf Q8_0
```

#### 3.2 進階量化選項（使用重要性矩陣）
```bash
# 生成重要性矩陣（可選，提升量化品質）
./build/bin/llama-imatrix -m ./models/llama-3.2-1b-f16.gguf \
  -f calibration_data.txt \
  -o ./models/llama-3.2-1b.imatrix

# 使用重要性矩陣進行量化
./build/bin/llama-quantize ./models/llama-3.2-1b-f16.gguf \
  ./models/llama-3.2-1b-q4_k_m_imatrix.gguf Q4_K_M \
  --imatrix ./models/llama-3.2-1b.imatrix
```

### 4. 測試與比較

#### 4.1 模型大小比較
```bash
ls -lh models/*.gguf
```

預期結果：
- F16: ~2.4GB
- Q8_0: ~1.3GB  
- Q5_K_M: ~900MB
- Q4_K_M: ~700MB
- Q4_0: ~650MB

#### 4.2 推理速度測試
```bash
# 使用 llama-bench 測試各模型性能
./build/bin/llama-bench -m models/llama-3.2-1b-f16.gguf
./build/bin/llama-bench -m models/llama-3.2-1b-q4_k_m.gguf
./build/bin/llama-bench -m models/llama-3.2-1b-q8_0.gguf
```

#### 4.3 困惑度測試（品質評估）
```bash
# 準備測試文本
echo "Your test text here..." > test.txt

# 測試各模型的困惑度
./build/bin/llama-perplexity -m models/llama-3.2-1b-f16.gguf -f test.txt
./build/bin/llama-perplexity -m models/llama-3.2-1b-q4_k_m.gguf -f test.txt
```

#### 4.4 實際推理測試
```bash
# F16 模型
./build/bin/llama-cli -m models/llama-3.2-1b-f16.gguf \
  -p "Once upon a time" -n 50

# Q4_K_M 模型
./build/bin/llama-cli -m models/llama-3.2-1b-q4_k_m.gguf \
  -p "Once upon a time" -n 50
```

### 5. GPU 加速測試
```bash
# 測試 GPU 加速效能
./build/bin/llama-cli -m models/llama-3.2-1b-q4_k_m.gguf \
  -p "Write a short story" -n 100 \
  -ngl 35  # 將所有層載入 GPU
```

## 量化格式說明

| 格式 | 位元數 | 相對大小 | 品質 | 用途建議 |
|------|--------|----------|------|----------|
| F16 | 16 | 100% | 最高 | 基準測試、轉換源 |
| Q8_0 | 8 | ~54% | 極高 | 高品質需求 |
| Q6_K | 6 | ~41% | 很高 | 品質優先 |
| Q5_K_M | 5 | ~37% | 高 | 平衡選擇 |
| Q4_K_M | 4 | ~29% | 中高 | 推薦使用 |
| Q4_0 | 4 | ~27% | 中 | 記憶體受限 |
| Q3_K_M | 3 | ~22% | 中低 | 極度受限環境 |
| Q2_K | 2 | ~15% | 低 | 實驗用途 |

## 預期學習成果

1. **理解量化原理**：不同量化方法如何影響模型大小和品質
2. **工具鏈熟悉**：掌握 llama.cpp 的轉換和量化工具
3. **性能評估**：學會評估量化模型的品質和速度
4. **最佳實踐**：找出適合不同場景的量化配置

## 注意事項

1. **授權問題**：Llama 3.2 需要接受 Meta 的授權條款
2. **記憶體需求**：轉換 F16 模型需要約 8GB RAM
3. **儲存空間**：完整實驗需要約 10GB 硬碟空間
4. **CUDA 支援**：確保 CUDA 已正確設置以使用 GPU 加速

## 後續實驗建議

1. 嘗試其他模型架構（Qwen, Mistral 等）
2. 測試混合精度量化
3. 探索 LoRA 適配器的量化
4. 比較不同硬體上的性能差異