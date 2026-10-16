# 如何貢獻 🤗 LeRobot

歡迎所有人參與貢獻，我們重視每個人的付出。寫程式並不是幫助社群的唯一方式，回答問題、協助他人、主動聯繫，以及改善說明文件，都是非常有價值的貢獻。

無論你選擇以哪種方式貢獻，請遵守我們的[行為準則](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)與 [AI 政策](https://github.com/huggingface/lerobot/blob/main/AI_POLICY.md)。

## 貢獻方式

你可以透過多種方式貢獻：

- **修正問題：** 修正錯誤，或改善現有的程式碼。
- **新功能：** 開發新的功能。
- **擴充：** 實作新的模型／策略（policy）、機器人或模擬環境，以及將資料集上傳到 Hugging Face Hub。
- **說明文件：** 改善範例、指南與 docstring。
- **意見回饋：** 回報錯誤，或提出想要的新功能。

如果不確定從哪裡開始，歡迎加入我們的 [Discord 頻道](https://discord.gg/q8Dzzpym3f)。

## 開發環境設定

若要貢獻程式碼，你需要先設定開發環境。

### 1. Fork 與 clone

在 GitHub 上 fork 這個儲存庫，然後 clone 你的 fork：

```bash
git clone https://github.com/<your-handle>/lerobot.git
cd lerobot
git remote add upstream https://github.com/huggingface/lerobot.git
```

### 2. 安裝環境

請依照我們的[安裝指南](https://huggingface.co/docs/lerobot/installation)設定環境，並從原始碼安裝。

## 執行測試與品質檢查

### 程式碼風格（pre-commit）

安裝 `pre-commit` hook，在每次 commit 之前自動執行檢查：

```bash
pre-commit install
```

若要手動對所有檔案執行檢查：

```bash
pre-commit run --all-files
```

### Docstring

API 參考文件是由 `src/lerobot/` 中的 docstring 產生的。如果你新增或修改任何公開的內容，請遵守 [docstring 規範](https://huggingface.co/docs/lerobot/writing_docstrings)：渲染器會解析這個格式，CI 也會檢查。

### 執行測試

我們使用 `pytest`。首先，請安裝 **git-lfs**，確保你有測試所需的檔案：

```bash
git lfs install
git lfs pull
```

執行完整的測試（可能需要安裝額外套件）：

```bash
pytest -sv ./tests
```

或在開發時只執行特定的測試檔：

```bash
pytest -sv tests/test_specific_feature.py
```

## 提交 issue 與 pull request

請使用範本，裡面有必填欄位與範例。

- **Issue：** 請依照 [issue 範本](https://github.com/huggingface/lerobot/blob/main/.github/ISSUE_TEMPLATE/bug-report.yml)填寫。
- **Pull request：** rebase 到 `upstream/main`，使用名稱有意義的分支（不要直接在 `main` 上開發），在本機執行 `pre-commit` 與測試，並依照 [PR 範本](https://github.com/huggingface/lerobot/blob/main/.github/PULL_REQUEST_TEMPLATE.md)填寫。

> [!IMPORTANT]
> 社群審查政策：為了擴大我們的處理量能並營造協作的環境，我們要求貢獻者在自己的 PR 獲得處理之前，先審查至少一個其他人尚未合併的 PR。這份共同的責任能讓審查能量加倍，也能讓每個人的程式碼更快被合併！

提交 PR 並完成同儕審查後，LeRobot 團隊的成員就會審查你的貢獻。

感謝你為 LeRobot 做出貢獻！
