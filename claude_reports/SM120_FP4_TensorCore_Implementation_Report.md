# SM120 原生 FP4 Tensor Core 實作評估與建議（可交付版）

**版本**：v1.0  
**更新日期**：2025-08-11（Asia/Taipei）  
**目標讀者**：系統軟體工程師、效能工程師、ML 平台/框架維護者、CI 維運人員  
**適用硬體**：GeForce Blackwell / RTX 50 系列（SM120）

---

## 1) 結論摘要（TL;DR）

- 報告大方向**合理**：以 CUTLASS 4.1 打通 SM120 的 **FP4(E2M1)** 原生 Tensor Core 路徑是正確策略。  
- **推薦工具鏈**：**CUDA 13.0（首選）**；若受限可採 **CUDA 12.9**。驅動需 **R580 系列或更新**。  
- **建置重點**：
  - `-gencode arch=compute_120,code=sm_120`  
  - `-gencode arch=compute_120,code=compute_120`（同時打包 SASS 與 PTX）  
  - 以 CUTLASS 4.1 Blackwell（GeForce）範例為藍本，優先走 **NVFP4/MXFP4** block-scaled 路徑。  
- **需修正**：
  - 改用 CUTLASS 既有的能力偵測巨集/traits（避免自定義 `CUTE_ARCH_F8F6F4_MMA_ENABLED` 這類不穩定巨集）。
  - 將「ptxas 錯誤＝驗證成功」修正為「語法被解析，但工具鏈/目標組態未匹配」。
  - 效能表列標示為**目標/預估**，並補上實測計畫（模型、張量形狀、記憶體佈局、KV cache 策略、block-scaling/反量化開銷）。

---

## 2) 合理性評估—逐點核對

### 2.1 架構與資料型別
- Blackwell 第五代 Tensor Core 在 GeForce 分支（SM120）提供 **FP4** 原生支援。  
- CUTLASS 4.1 已納入 Blackwell 子位寬型別（包含 FP4/MXFP4/NVFP4）與相對應範例；型別如 `cutlass::float_e2m1_t` 與相關 unpack/load/store 工具可直接對應。

### 2.2 工具鏈與 PTX 指令族
- Blackwell 新增 **`.kind::f8f6f4`** 家族用於 FP8/FP6/FP4 相關 UMMA 指令。  
- 若 `ptxas` 出現 `Feature '.kind::f8f6f4' not supported on .target 'sm_120'`，代表**語法被觸發**但工具鏈版本、驅動或 gencode 目標未對齊；**不能視為已完成可用的驗證**。

### 2.3 報告中合理之處
- 以 CUTLASS 4.1 整合、在 SM120 走 MXFP4/FP4 路徑，並保留 INT8 後備路徑的策略正確。  
- 使用 CMake/條件編譯把向量點積替換為 FP4 路徑，整體模組化與風險控制合理。

### 2.4 需修正/保留彈性的細節
1. **巨集名稱**：採用 CUTLASS/CuTe 既有能力偵測巨集或 traits（例如基於 SM 版號與 OP 能力），避免自創名稱造成誤判。  
2. **型別/類名對應**：優先比照 CUTLASS Blackwell 示例與 `tcgen05` 路徑的 Collective Builder 與 pipeline 模板；避免使用未公開或不穩定命名（例如手寫 `SM120_16x8x32_TN<...>` 類別）。  
3. **驗證敘述**：把「ptxas 錯誤」改述為「*解析成功但組譯不支持*」，並新增 **CUDA 13.0 + R580** 下可**組譯與執行**的實測欄位。  
4. **效能數據**：4–8× 屬理想上限，易受記憶體重排、KV cache、block-scaling/反量化與張量形狀影響；需以**實卡**與**實際模型**補強。

---

## 3) 建議的工具鏈與版本（SM120 / RTX 50 系列）

### 3.1 CUDA Toolkit
- **首選：CUDA 13.0**  
  - 延續 12.8/12.9 的 Blackwell 支援並加強；對 FP4/FP6/FP8 工具鏈相容性最佳。  
- **可行下限：CUDA 12.9**  
  - 引入 Blackwell 重要特性與 `.kind::f8f6f4` 相關支援；若受限於環境可暫用。  
- **僅建置但特性可能不全：CUDA 12.8**  
  - Blackwell 的初次支援版本；不建議作為長期基準。

### 3.2 顯示驅動（Driver）
- **需求**：**R580 系列或更新**（Linux/Windows 對應分支）。  
- 舊版驅動可能導致 `ptxas` 或執行時不支援 `.kind::f8f6f4`。

### 3.3 CUTLASS / 編譯器
- **CUTLASS**：建議 **v4.1.x**。  
- **NVCC/gencode**：
  ```bash
  nvcc -gencode arch=compute_120,code=sm_120        -gencode arch=compute_120,code=compute_120        -O3 -std=c++17
  ```
- **CMake 提示**：確保 `CMAKE_CUDA_ARCHITECTURES=120-real;120`（或同等設定），並以 Blackwell（GeForce）示例為模板。

---

## 4) 實作與驗證指引

### 4.1 介面與型別
- 以 CUTLASS 提供的 **NVFP4/MXFP4** 型別與 **unpack/load** 工具替換現行 INT8/FP8 路徑中的內核資料流。  
- 依 `tcgen05`/TMA 流水線範式組裝：
  - TMA 載入 → shared memory 佈局/轉置 → UMMA（FP4） → epilogue（scale/activation/反量化）。

### 4.2 條件編譯與能力偵測
- 使用 CUTLASS/CuTe 既有的 **SM 能力巨集或 traits**（例如 `__CUDA_ARCH__ >= 1200` 或 CUTLASS 的 OP 能力標記），避免自定義巨集造成交叉編譯誤判。

### 4.3 例外與回退
- 若偵測工具鏈或驅動不支持 FP4 UMMA：
  - 回退至 **INT8** 或 **FP8**（依模型精度/吞吐需求選擇）。
  - 在 CI 以 **feature probe**（嘗試小型 kernel 組譯+執行）決定路徑。

### 4.4 效能評測計畫（建議最小集合）
- **模型**：Llama/Whisper/UNet（三種算子結構不同）。  
- **張量形狀**：小/中/大三檔（含長序列 KV cache）。  
- **指標**：吞吐（tokens/s 或 samples/s）、延遲（p50/p99）、顯存佔用、能效（samples/J）。  
- **實驗因子**：
  - block-scaling 粒度（per-channel / per-tile）。
  - 反量化位置（epilogue/外層 fusion）。
  - 佈局：RowMajor vs. ColumnMajor、swizzle、TMA tile 設計。

---

## 5) 對原始報告的建議修改（可直接套用）

1. **驅動需求**：將「575.64+」改為 **R580+**。  
2. **條件編譯巨集**：以 CUTLASS 既有 **SM120 能力巨集/traits** 或 `__CUDA_ARCH__` 判斷取代自定義巨集。  
3. **技術驗證敘述**：
   - 將「ptxas 錯誤＝驗證成功」改為「*語法被解析但未完成彙編*」。  
   - 新增以 **CUDA 13.0 + R580** 成功**組譯與執行**的實測結果欄位（包含最小 kernel 與矩陣維度）。  
4. **效能表**：標註**目標/預估**，並附上**實測方法**與**環境**（GPU 型號、Toolkit/Driver 版本、時脈/功耗設定）。

---

## 6) 風險與緩解

- **工具鏈碎裂**：分支混用（12.8/12.9/13.0）導致行為差異 → 建議統一到 **13.0 + R580**。  
- **PTX/SASS 相容性**：缺少 `code=compute_120` 回退時，未來驅動行為可能變動 → **同時打包 PTX**。  
- **佈局/資料流瓶頸**：即便算力提升，未優化的資料移動會吞噬收益 → 以 **TMA + shared 轉置** 與 **fused epilogue** 緩解。

---

## 7) 快速安裝與編譯備忘

```bash
# 以 Linux 為例（需 R580+ 驅動）
# 1) 安裝 CUDA 13.0（或 12.9）
# 2) 取得 CUTLASS 4.1.x 並啟用 Blackwell 示例

# 編譯 flags（示例）
nvcc -gencode arch=compute_120,code=sm_120      -gencode arch=compute_120,code=compute_120      -O3 -std=c++17 your_kernel.cu -o your_kernel.out
```

---

## 8) 參考（標題級，供檢索）
- NVIDIA CUDA 13.0 Release Notes / Toolkit Document
- NVIDIA CUDA 12.9 Release Notes / Toolkit Document
- NVIDIA R580 Driver Branch Notes
- NVIDIA Blackwell Architecture/Tuning Guide（GeForce/SM120）
- CUTLASS 4.1.x 文件與 Blackwell 示例（tcgen05、NVFP4/MXFP4）
- PTX ISA（含 `.kind::f8f6f4` 指令家族）

---

*備註：本文為內部可交付稿，不含外部平台專用引用符號；如需完整文獻連結清單，請告知將另附外部參考附錄。*
