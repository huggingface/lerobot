# 如何貢獻至 🤗 LeRobot

歡迎所有人貢獻，我們重視每個人的貢獻。撰寫程式碼並非協助社群的唯一方式。回答問題、協助他人、積極聯繫，以及改善文件說明，都是極具價值的貢獻。

無論您選擇以何種方式貢獻，請遵守我們的[行為準則（code of conduct）](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)與 [AI 政策](https://github.com/huggingface/lerobot/blob/main/AI_POLICY.md)。

## 貢獻方式（ways to contribute）

您可以透過多種方式做出貢獻：

- **修復問題（fixing issues）：** 解決錯誤或改善現有的程式碼。
- **新功能（new features）：** 開發新功能。
- **擴充（extend）：** 實作新的模型／策略（models/policies）、機器人或模擬環境（robots or simulation environments）以及上傳資料集（datasets）至 Hugging Face Hub。
- **文件（documentation）：** 改善範例、指南與文件字串（docstrings）。
- **意見回饋（feedback）：** 提交與錯誤（bugs）或所需新功能（new features）相關的工單（tickets）。

若您不確定從何開始，歡迎加入我們的 [Discord Channel](https://discord.gg/q8Dzzpym3f)。

## 開發環境設定（development setup）

若要貢獻程式碼，您需要建立一個開發環境。

### 1. Fork 與 Clone

在 GitHub 上 fork 儲存庫（repository），然後 clone 您的 fork：

```bash
git clone https://github.com/<your-handle>/lerobot.git
cd lerobot
git remote add upstream https://github.com/huggingface/lerobot.git
```

### 2. 環境安裝（environment installation）

請參閱我們的[安裝指南](https://huggingface.co/docs/lerobot/installation)，進行環境設定並以原始碼安裝（installation from source）。

## 執行測試 & 品質檢查（running tests & quality checks）

### 程式碼風格（code style, pre-commit）

安裝 `pre-commit` hooks，以在提交前自動執行檢查：

```bash
pre-commit install
```

若要對所有檔案手動執行檢查：

```bash
pre-commit run --all-files
```

### 執行測試（running tests）

我們使用 `pytest`。首先，請安裝 **git-lfs** 以確保您擁有所需的測試構件（test artifacts）：

```bash
git lfs install
git lfs pull
```

執行完整測試套件（run the full suite，可能需要安裝額外套件 extras）：

```bash
pytest -sv ./tests
```

或在開發期間執行特定測試檔案：

```bash
pytest -sv tests/test_specific_feature.py
```

## 提交 Issues & Pull Requests

請使用範本（template）來取得必填欄位（required fields）與範例。

- **Issues:** 請遵循 [工單模板](https://github.com/huggingface/lerobot/blob/main/.github/ISSUE_TEMPLATE/bug-report.yml)。
- **Pull requests:** 以 `upstream/main` 進行 rebase，使用具描述性的分支名稱（請勿直接在 `main` 上工作），在本地端執行 `pre-commit` 與測試，並遵循 [PR 範本](https://github.com/huggingface/lerobot/blob/main/.github/PULL_REQUEST_TEMPLATE.md)。

> [!IMPORTANT]
> 社群審查政策（community review policy）：為了擴大我們的工作效能並培養協作環境，我們要求貢獻者在自己的 PR 得到關注之前，先審查至少一個其他人的開放 PR。這項共同責任能倍增我們的審查能量，並幫助每個人的程式碼更快被合併（merged）！

在您提交 PR 並完成同儕審查（peer review）後，LeRobot 團隊的成員將會審查您的貢獻。

感謝您為 LeRobot 做出的貢獻！
