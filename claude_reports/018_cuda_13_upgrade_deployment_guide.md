# CUDA 13.0 升級部署指南

**日期**: 2025年8月11日  
**目的**: 升級至 CUDA 13.0 以支援 RTX 5070 原生 FP4 tensor core  
**狀態**: 📋 部署規劃文檔

## 🎯 執行摘要

CUDA 13.0 已正式發布，完整支援 Blackwell SM120 架構的原生 FP4 指令。本指南提供 Docker 環境下的完整升級策略，確保我們的原生 FP4 tensor core 實現能夠成功運行。

## 🐳 Docker 環境升級策略分析

### **方案比較**

| 升級方案 | 優勢 | 劣勢 | 推薦度 |
|----------|------|------|--------|
| **Docker 內升級** | ✅ 環境隔離<br>✅ 快速測試<br>✅ 易於回滾 | ❌ 驅動依賴宿主機<br>❌ 權限複雜 | 🟡 條件性推薦 |
| **宿主機升級** | ✅ 完整控制<br>✅ 驅動直接管理<br>✅ 性能最佳 | ❌ 影響系統<br>❌ 回滾困難 | 🟢 **強烈推薦** |
| **新容器環境** | ✅ 乾淨環境<br>✅ 版本隔離<br>✅ 並行測試 | ❌ 重新配置<br>❌ 數據遷移 | 🟡 次選方案 |

### **🎯 推薦方案: 宿主機升級**

**原因分析**:
1. **驅動依賴**: NVIDIA 驅動必須在宿主機安裝，Docker 容器內無法獨立管理
2. **硬體訪問**: SM120 FP4 指令需要完整的驅動支援
3. **效能考量**: 原生 tensor core 需要最佳的硬體訪問路徑

## 📋 宿主機升級完整指南

### **階段 1: 環境檢查與備份**

#### 1.1 檢查當前環境
```bash
# 在 Docker 外執行 (宿主機)
nvidia-smi
nvcc --version
lsb_release -a  # 檢查系統版本
```

#### 1.2 備份當前工作
```bash
# 備份 Docker 容器工作 (在容器內執行)
cd /workspace/llama.cpp
tar -czf /tmp/llama_cpp_work_backup.tar.gz \
    ggml/src/ggml-cuda/cutlass_mxfp4_native.cuh \
    ggml/src/ggml-cuda/vecdotq.cuh \
    ggml/CMakeLists.txt \
    ggml/src/ggml-cuda/CMakeLists.txt \
    claude_reports/ \
    vendor/cutlass/

# 複製到宿主機 (在宿主機執行)
docker cp <container_id>:/tmp/llama_cpp_work_backup.tar.gz ./
```

### **階段 2: 宿主機 CUDA 13.0 升級**

#### 2.1 移除舊版本 CUDA (可選)
```bash
# Ubuntu/Debian
sudo apt-get --purge remove "*cuda*" "*cublas*" "*cufft*" "*cufile*" "*curand*"
sudo apt-get autoremove

# 清理舊版本
sudo rm -rf /usr/local/cuda*
```

#### 2.2 安裝 CUDA 13.0
```bash
# Ubuntu 22.04 LTS (推薦方法)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update

# 安裝完整 CUDA Toolkit 13.0
sudo apt-get install cuda-toolkit-13-0

# 設定環境變數
echo 'export PATH=/usr/local/cuda-13.0/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CUDA_HOME=/usr/local/cuda-13.0' >> ~/.bashrc
source ~/.bashrc
```

#### 2.3 驅動升級
```bash
# 檢查當前驅動版本
nvidia-smi | grep "Driver Version"

# 如果 < 580.65.06，執行升級
sudo apt update
sudo apt install nvidia-driver-580

# 重啟系統
sudo reboot
```

### **階段 3: Docker 環境重新配置**

#### 3.1 使用 NVIDIA Container Toolkit
```bash
# 安裝 NVIDIA Container Runtime (如果尚未安裝)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

#### 3.2 啟動支援 CUDA 13.0 的容器
```bash
# 使用 NVIDIA 官方 CUDA 13.0 映像
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    nvidia/cuda:13.0-devel-ubuntu22.04 \
    bash

# 或繼續使用現有容器並映射 CUDA 13.0
docker run --gpus all -it --rm \
    -v /usr/local/cuda-13.0:/usr/local/cuda \
    -v $(pwd):/workspace \
    <your_current_image> \
    bash
```

### **階段 4: 驗證與測試**

#### 4.1 驗證 CUDA 13.0 環境
```bash
# 在容器內驗證
nvcc --version | grep "release 13.0"
nvidia-smi

# 檢查 SM120 支援
deviceQuery  # 如果可用
# 或
nvidia-smi --query-gpu=compute_cap --format=csv
```

#### 4.2 測試 CUTLASS FP4 編譯
```bash
# 恢復備份的工作
cd /workspace
tar -xzf llama_cpp_work_backup.tar.gz

# 測試編譯
cd llama.cpp
cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_CUDA_CUTLASS_FP4=ON \
    -DCMAKE_CUDA_ARCHITECTURES="120-real;120" \
    -DCMAKE_BUILD_TYPE=Release

# 關鍵測試：之前的 ptxas 錯誤應該消失
cmake --build build --config Release -j 2
```

## 📊 預期結果與故障排除

### **成功指標**
- ✅ `nvcc --version` 顯示 `release 13.0`
- ✅ `nvidia-smi` 顯示驅動 ≥580.65.06
- ✅ CUTLASS 編譯無 ptxas 錯誤
- ✅ 生成 `build/bin/llama-cli` 等執行檔

### **常見問題與解決**

#### 問題 1: Docker 內無法看到 GPU
```bash
# 解決方案：檢查 NVIDIA Container Runtime
docker run --gpus all nvidia/cuda:13.0-base-ubuntu22.04 nvidia-smi

# 如果失敗，重新安裝 nvidia-container-toolkit
sudo apt-get install --reinstall nvidia-container-toolkit
sudo systemctl restart docker
```

#### 問題 2: CUDA 版本不匹配
```bash
# 檢查實際載入的 CUDA 版本
ls /usr/local/cuda*/version.json
cat /usr/local/cuda/version.json

# 如果有多個版本，設定正確的符號連結
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-13.0 /usr/local/cuda
```

#### 問題 3: 編譯時找不到 CUTLASS
```bash
# 確保 CUTLASS 子模組存在
cd /workspace/llama.cpp
git submodule update --init --recursive
ls vendor/cutlass/
```

## 🎯 關鍵時程規劃

### **預估執行時間**
- **系統準備**: 30 分鐘
- **CUDA 13.0 下載安裝**: 45-60 分鐘  
- **驅動升級與重啟**: 30 分鐘
- **Docker 重新配置**: 20 分鐘
- **編譯測試**: 15-20 分鐘
- **總計**: **2.5-3 小時**

### **里程碑檢查點**
1. ✅ **CheckPoint 1**: `nvcc --version` = 13.0
2. ✅ **CheckPoint 2**: `nvidia-smi` 正常顯示 RTX 5070
3. ✅ **CheckPoint 3**: Docker 容器能訪問 GPU
4. ✅ **CheckPoint 4**: CUTLASS 編譯無錯誤
5. ✅ **CheckPoint 5**: llama.cpp 成功建置

## 🚀 升級後的驗證測試

### **功能驗證**
```bash
# 1. 基本編譯測試
cmake --build build --config Release

# 2. CUTLASS FP4 路徑測試
./build/bin/llama-cli --help | grep -i cutlass  # 如果有相關選項

# 3. GPU 記憶體使用測試
nvidia-smi dmon -s mu -c 5  # 監控 GPU 使用率
```

### **效能基準測試** (升級成功後執行)
```bash
# 準備測試模型 (例如)
wget https://huggingface.co/microsoft/DialoGPT-small/resolve/main/pytorch_model.bin

# MXFP4 效能測試 (具體命令待確認)
./build/bin/llama-bench -m model.gguf -t 4 --mxfp4

# 與 FP16 對比測試
./build/bin/llama-bench -m model.gguf -t 4 --fp16
```

## ⚡ 快速執行摘要

### **立即執行清單**
1. 🔄 **離開 Docker 到宿主機**
2. 📦 **備份當前工作**: `docker cp container:/workspace/llama.cpp ./`
3. 🛠 **升級 CUDA**: 按照階段 2 指令執行
4. 🔧 **重啟系統**: `sudo reboot`  
5. 🐳 **重新配置 Docker**: 使用 CUDA 13.0 映像
6. ✅ **驗證測試**: CUTLASS 編譯成功

### **成功標準**
- ptxas 錯誤消失
- 原生 FP4 代碼成功編譯
- RTX 5070 SM120 功能完全啟用
- 準備體驗 4-8x MXFP4 推理加速！

---

**建議**: 立即開始宿主機升級，這是啟用真正原生 FP4 tensor core 加速的關鍵步驟。升級完成後，我們將見證 llama.cpp 歷史上第一次真正的硬體 FP4 加速！

**最後更新**: 2025年8月11日  
**預期完成**: CUDA 13.0 升級後即刻可用