# 如何为 🤗 LeRobot 做出贡献

我们欢迎每个人做出贡献，并重视每个人的贡献。代码并不是帮助社区的唯一方式。回答问题、帮助他人、分享交流以及改进文档都非常有价值。

无论您选择以何种方式做出贡献，请务必尊重我们的[行为准则](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md)和[AI 政策](https://github.com/huggingface/lerobot/blob/main/AI_POLICY.md)。

## 贡献方式

您可以通过多种方式做出贡献：

- **修复问题：** 解决 bug 或改进现有代码。
- **新功能：** 开发新功能。
- **扩展：** 实现新的模型/策略、机器人或仿真环境，并将数据集上传到 Hugging Face Hub。
- **文档：** 改进示例、指南和文档字符串。
- **反馈：** 提交与 bug 或期望的新功能相关的工单。

如果您不确定从哪里开始，请加入我们的 [Discord 频道](https://discord.gg/q8Dzzpym3f)。

## 开发环境设置

要贡献代码，您需要设置开发环境。

### 1. Fork 和克隆

在 GitHub 上 fork 仓库，然后克隆您的 fork：

```bash
git clone https://github.com/<您的用户名>/lerobot.git
cd lerobot
git remote add upstream https://github.com/huggingface/lerobot.git
```

### 2. 环境安装

请按照我们的[安装指南](https://huggingface.co/docs/lerobot/installation)进行环境设置和从源码安装。

## 运行测试和质量检查

### 代码风格（Pre-commit）

安装 `pre-commit` 钩子，以便在提交前自动运行检查：

```bash
pre-commit install
```

要手动对所有文件运行检查：

```bash
pre-commit run --all-files
```

### 运行测试

我们使用 `pytest`。首先，通过安装 **git-lfs** 确保您拥有测试文件：

```bash
git lfs install
git lfs pull
```

运行完整的测试套件（这可能需要安装额外的依赖）：

```bash
pytest -sv ./tests
```

或在开发期间运行特定的测试文件：

```bash
pytest -sv tests/test_specific_feature.py
```

## 提交问题和拉取请求

使用模板填写必填字段和示例。

- **问题：** 遵循[工单模板](https://github.com/huggingface/lerobot/blob/main/.github/ISSUE_TEMPLATE/bug-report.yml)。
- **拉取请求：** 基于 `upstream/main` 进行 rebase，使用描述性的分支名称（不要在 `main` 上工作），在本地运行 `pre-commit` 和测试，并遵循[PR 模板](https://github.com/huggingface/lerobot/blob/main/.github/PULL_REQUEST_TEMPLATE.md)。

> [!重要]
> 社区审查政策：为了帮助扩大我们的影响力并培养协作环境，我们要求贡献者在自己的 PR 获得关注之前，至少审查另一个人的开放 PR。这种共同责任将成倍增加我们的审查能力，并帮助每个人的代码更快合并！

提交 PR 并完成同行评审后，LeRobot 团队的成员将审查您的贡献。

感谢您为 LeRobot 做出贡献！