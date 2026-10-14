# 如何为 🤗 LeRobot 做贡献

每个人都欢迎贡献，我们重视每个人的贡献。代码不是帮助社区的唯一方式。回答问题、帮助他人、向外联系和改进文档都同样有价值。

无论你选择哪种方式贡献，请务必遵守我们的 [行为准则](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md) 和我们的 [AI 政策](https://github.com/huggingface/lerobot/blob/main/AI_POLICY.md)。

## 贡献方式

你可以通过多种方式贡献：

- **修复问题：** 解决 bug 或改进现有代码。
- **新功能：** 开发新功能。
- **扩展：** 实现新模型/策略、机器人或仿真环境，并将数据集上传到 Hugging Face Hub。
- **文档：** 改进示例、指南和文档字符串。
- **反馈：** 提交与 bug 或期望的新功能相关的工单。

如果你不确定从哪里开始，请加入我们的 [Discord 频道](https://discord.gg/q8Dzzpym3f)。

## 开发设置

要贡献代码，你需要设置开发环境。

### 1. Fork 和克隆

在 GitHub 上 fork 仓库，然后克隆你的 fork：

```bash
git clone https://github.com/<你的用户名>/lerobot.git
cd lerobot
git remote add upstream https://github.com/huggingface/lerobot.git
```

### 2. 环境安装

请按照我们的 [安装指南](https://huggingface.co/docs/lerobot/installation) 进行环境设置和从源代码安装。

## 运行测试和质量检查

### 代码风格（Pre-commit）

安装 `pre-commit` 钩子，在提交前自动运行检查：

```bash
pre-commit install
```

要在所有文件上手动运行检查：

```bash
pre-commit run --all-files
```

### 运行测试

我们使用 `pytest`。首先，确保你通过安装 **git-lfs** 拥有测试工件：

```bash
git lfs install
git lfs pull
```

运行完整测试套件（这可能需要安装额外依赖）：

```bash
pytest -sv ./tests
```

或在开发过程中运行特定的测试文件：

```bash
pytest -sv tests/test_specific_feature.py
```

## 提交问题和拉取请求

使用模板填写必填字段和示例。

- **问题：** 遵循 [工单模板](https://github.com/huggingface/lerobot/blob/main/.github/ISSUE_TEMPLATE/bug-report.yml)。
- **拉取请求：** 基于 `upstream/main` rebase，使用描述性的分支（不要在 `main` 上工作），在本地运行 `pre-commit` 和测试，并遵循 [PR 模板](https://github.com/huggingface/lerobot/blob/main/.github/PULL_REQUEST_TEMPLATE.md)。

> [!IMPORTANT]
> 社区审查政策：为了帮助扩大我们的努力并培养协作环境，我们要求贡献者在自己的 PR 获得关注之前至少审查一个其他人的开放 PR。这种共同责任可以将我们的审查能力成倍增加，并帮助每个人的代码更快合并！

提交 PR 并完成同行审查后，LeRobot 团队的成员将审查你的贡献。

感谢你为 LeRobot 做出贡献！
