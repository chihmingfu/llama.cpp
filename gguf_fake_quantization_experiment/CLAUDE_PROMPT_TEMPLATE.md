# Claude Prompt Template for GGUF Model Quantization Experiments

## 完整Prompt模板

```
You are an expert in machine learning model quantization research, specifically working with GGUF format models and llama.cpp. I need you to help me conduct a systematic quantization experiment.

## Experiment Context
- **Target**: [描述具體要量化的權重類型，如 "FFN norm weights", "attention weights", "embedding weights"]
- **Quantization Type**: [指定量化類型，如 "BF16 fake quantization", "INT8 quantization", "4-bit quantization"]
- **Models**: [列出要測試的模型，如 "TinyLlama 1.1B", "Llama-3.2-1B"]
- **Evaluation Metrics**: [指定評估指標，如 "perplexity", "conversation quality", "numerical differences"]

## Required Approach
1. **Systematic Planning**: Create detailed experiment plan with step-by-step methodology
2. **Tool Development**: Develop quantization tools with proper validation
3. **Progressive Testing**: Start with single layer, then expand gradually
4. **Comprehensive Analysis**: Include numerical analysis, model comparison, and performance evaluation
5. **Thorough Documentation**: Create detailed logs, reports, and reproducible workflows

## Technical Requirements
- Use existing llama.cpp and gguf-py infrastructure
- Implement proper error handling and validation
- Create tools with default parameters for easy testing
- Include precision analysis to detect already-quantized weights
- Verify tool correctness through multi-model testing
- Use proper file organization and documentation structure

## Key Principles from Previous Experience
1. **Verify Tool Correctness**: Always validate quantization tools with known test cases
2. **Check Weight Precision**: Analyze existing model precision before assuming quantization effects
3. **Progressive Verification**: If results seem "too good to be true", investigate systematically
4. **Multi-Model Validation**: Test on different models to confirm tool functionality
5. **Complete Documentation**: Record all commands, decisions, and findings for reproducibility

## Expected Deliverables
- Quantization implementation tool
- Precision analysis scripts
- Complete experimental results
- Comparison reports
- Usage documentation
- GitHub-ready project structure

## Workflow Methodology
1. **Environment Setup**: Verify dependencies and create project structure
2. **Baseline Analysis**: Examine target model's existing precision distribution
3. **Tool Development**: Create and validate quantization implementation
4. **Systematic Testing**: Execute planned experiments with proper controls
5. **Results Verification**: Cross-validate findings with multiple approaches
6. **Documentation**: Create comprehensive reports and usage guides

Please follow this systematic approach and maintain detailed logs throughout the process. If you encounter unexpected results, investigate thoroughly before concluding.
```

## 使用示例

### FFN Norm BF16 Fake Quantization (我們剛完成的實驗)
```
Target: FFN norm weights  
Quantization Type: BF16 fake quantization
Models: TinyLlama 1.1B Chat v1.0 (Q4_K_M), Llama-3.2-1B-Instruct-f16
Evaluation Metrics: numerical differences, perplexity, conversation quality
```

### Attention Weights INT8 Quantization (未來實驗示例)
```
Target: Attention weights (Q, K, V projection matrices)
Quantization Type: INT8 fake quantization  
Models: Llama-3.2-1B-Instruct-f16, Qwen2-1.5B-Instruct
Evaluation Metrics: attention pattern analysis, downstream task performance, numerical precision loss
```

### Embedding Weights 4-bit Quantization (未來實驗示例)
```
Target: Token embedding weights
Quantization Type: 4-bit fake quantization
Models: Multiple architectures and sizes
Evaluation Metrics: token representation quality, semantic similarity preservation, vocabulary coverage
```

## 關鍵提醒事項

### 基於實際經驗的重要提示
1. **現代模型可能已經優化**：許多GGUF模型的norm權重已經是BF16精度
2. **工具驗證必不可少**：必須找到具有完整精度的權重來驗證工具正確性
3. **逐步驗證方法**：從簡單測試開始，逐步增加複雜性
4. **精度分析先行**：在量化前先分析目標權重的現有精度分布
5. **多模型交叉驗證**：使用不同模型驗證工具和方法的可靠性

### 常見陷阱避免
- ❌ 假設所有權重都有完整精度
- ❌ 忽略工具正確性驗證
- ❌ 過度依賴單一模型測試
- ❌ 缺乏系統性的實驗設計
- ❌ 未記錄完整的實驗過程

### 成功指標
- ✅ 工具通過多模型驗證
- ✅ 實驗結果可重現
- ✅ 發現具有技術價值
- ✅ 文檔完整且清晰
- ✅ 代碼結構良好且可重用

## Prompt優化建議

### 針對特定實驗類型的調整
1. **精度研究**：強調數值分析和統計驗證
2. **性能研究**：重點關注推理速度和內存使用
3. **質量研究**：側重模型輸出質量和任務性能
4. **工具開發**：注重代碼可重用性和通用性

### 模型特異性考慮
- **大模型**：考慮內存限制和處理時間
- **多模態模型**：包含視覺或音頻組件的特殊處理
- **特殊架構**：適應非標準模型結構
- **量化模型**：理解現有量化狀態

---

**使用方法**：複製以上prompt模板，根據具體實驗需求填入相應參數，然後直接與Claude開始對話。

**版本**：v1.0 - 基於FFN Norm BF16 Fake Quantization實驗經驗  
**更新日期**：2025-07-15