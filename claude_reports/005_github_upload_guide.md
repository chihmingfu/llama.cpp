# GitHub 上傳指南

## 當前狀態分析

根據 `git status` 的結果，您的工作區包含以下變更：

### 修改的文件
- `CLAUDE.md` - 已更新的構建和使用指南

### 新增的實驗報告
- `claude_reports/001_llama_quantization_experiment_plan.md` - 實驗計劃
- `claude_reports/002_mxfp4_quantization_info.md` - MXFP4 量化資訊
- `claude_reports/003_experiment_execution_log.md` - 完整實驗執行記錄
- `claude_reports/004_mxfp4_quantization_analysis.md` - MXFP4 問題分析報告

### 測試文件（建議不上傳）
- `wiki.test.raw`, `wiki.test.txt` - 測試資料文件
- `wikitext-2-raw-v1.zip` - 下載的資料集
- `=2.4.0` - 錯誤文件（需要清理）

## 安全設置指南

### 步驟 1：撤銷已洩露的 Token
1. 前往 [GitHub Settings](https://github.com/settings/tokens)
2. 找到並刪除剛才洩露的 token
3. 確認撤銷完成

### 步驟 2：生成新的 Personal Access Token
1. 前往 GitHub Settings > Developer settings > Personal access tokens
2. 點擊 "Generate new token (classic)"
3. 設置以下權限：
   - `repo` - 完整的倉庫控制權限
   - `workflow` - 如果需要更新 GitHub Actions
4. 複製生成的 token（只會顯示一次）

### 步驟 3：安全設置環境變數
```bash
# 將 YOUR_NEW_TOKEN 替換為實際的 token
export GITHUB_TOKEN="YOUR_NEW_TOKEN"

# 驗證設置
echo $GITHUB_TOKEN
```

## Git 操作流程

### 步驟 1：清理不需要的文件
```bash
# 刪除錯誤文件
rm "=2.4.0"

# 驗證清理結果
git status
```

### 步驟 2：設置 Git 配置（如果尚未設置）
```bash
# 設置您的 Git 身份
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 檢查當前配置
git config --list | grep user
```

### 步驟 3：選擇並添加要提交的文件
```bash
# 添加修改的 CLAUDE.md
git add CLAUDE.md

# 添加實驗報告目錄
git add claude_reports/

# 檢查 staged 的變更
git status
```

### 步驟 4：創建提交
```bash
git commit -m "Add comprehensive Llama 3.2 1B quantization experiments and analysis

- Update CLAUDE.md with enhanced build and usage instructions
- Add complete quantization experiment reports in claude_reports/
- Include performance benchmarks and quality analysis for Q4_0, Q4_K_M, Q5_K_M, Q8_0
- Document MXFP4_MOE quantization behavior and fallback mechanism
- Provide detailed perplexity testing results and recommendations

🤖 Generated with Claude Code (https://claude.ai/code)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

### 步驟 5：推送到 GitHub
```bash
# 推送到遠程倉庫
git push origin master

# 如果遇到認證問題，使用 token
git push https://$GITHUB_TOKEN@github.com/USERNAME/REPOSITORY.git master
```

## 可選：創建 .gitignore
為了避免將測試文件意外提交，建議創建或更新 `.gitignore`：

```bash
# 檢查是否已有 .gitignore
ls -la | grep gitignore

# 添加測試文件到 .gitignore (如果需要)
echo "# Test data files" >> .gitignore
echo "wiki.test.*" >> .gitignore
echo "*.zip" >> .gitignore
echo "# Temp files" >> .gitignore
echo "=*" >> .gitignore

# 提交 .gitignore 更新
git add .gitignore
git commit -m "Update .gitignore to exclude test data and temp files"
git push origin master
```

## 驗證上傳結果

### 步驟 1：檢查本地狀態
```bash
# 確認所有變更都已提交
git status

# 查看最近的提交
git log --oneline -3
```

### 步驟 2：檢查遠程倉庫
1. 前往您的 GitHub 倉庫頁面
2. 確認看到新的 commit
3. 檢查 `claude_reports/` 目錄是否正確顯示
4. 驗證 `CLAUDE.md` 的更新內容

## 安全注意事項

### ✅ 正確做法
- 使用環境變數存儲 token
- 定期更新和輪換 tokens
- 使用最小必要權限
- 不在代碼或聊天中硬編碼敏感信息

### ❌ 避免做法
- 在聊天、代碼或公共場所分享 tokens
- 使用過於寬泛的權限
- 長期使用同一個 token
- 在 commit 訊息中包含敏感信息

## 問題排除

### 認證問題
如果遇到 authentication failed：
```bash
# 方法 1: 使用 token 作為密碼
git push https://USERNAME@github.com/USERNAME/REPOSITORY.git master
# 輸入密碼時使用 token

# 方法 2: 更新遠程 URL
git remote set-url origin https://$GITHUB_TOKEN@github.com/USERNAME/REPOSITORY.git
```

### 大文件問題
如果遇到文件過大的警告：
```bash
# 檢查大文件
find . -size +50M -type f

# 從 staging 區移除大文件
git reset HEAD large_file.bin
```

## 後續建議

1. **定期備份**：考慮設置自動化的 git push 腳本
2. **分支策略**：對於未來的實驗，考慮使用特性分支
3. **標籤發布**：為重要的實驗結果打標籤
4. **文檔維護**：保持實驗報告的更新和組織

---

**創建日期**：2025-08-07  
**適用環境**：llama.cpp 倉庫  
**安全級別**：包含 GitHub token 處理指南