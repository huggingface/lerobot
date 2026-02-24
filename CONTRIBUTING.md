# How to contribute to 🤗 LeRobot

Everyone is welcome to contribute, and we value everybody's contribution. Code is not the only way to help the community. Answering questions, helping others, reaching out, and improving the documentation are immensely valuable.

Whichever way you choose to contribute, please be mindful to respect our [code of conduct](./CODE_OF_CONDUCT.md).

## Ways to Contribute

You can contribute in many ways:

- **Fixing issues:** Resolve bugs or improve existing code.
- **New features:** Develop new features.
- **Extend:** Implement new models/policies, robots, or simulation environments and upload datasets to the Hugging Face Hub.
- **Documentation:** Improve examples, guides, and docstrings.
- **Feedback:** Submit tickets related to bugs or desired new features.

If you are unsure where to start, join our [Discord Channel](https://discord.gg/q8Dzzpym3f).

## Development Setup

To contribute code, you need to set up a development environment.

### 1. Fork and Clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/<your-handle>/lerobot.git
cd lerobot
git remote add upstream https://github.com/huggingface/lerobot.git
```

### 2. Environment Installation

Please follow our [Installation Guide](./docs/source/installation.mdx) for the environment setup & installation from source.

## Running Tests & Quality Checks

### Code Style (Pre-commit)

Install `pre-commit` hooks to run checks automatically before you commit:

```bash
pre-commit install
```

To run checks manually on all files:

```bash
pre-commit run --all-files
```

### Running Tests

We use `pytest`. First, ensure you have test artifacts by installing **git-lfs**:

```bash
git lfs install
git lfs pull
```

Run the full suite (this may require extras installed):

```bash
pytest -sv ./tests
```

Or run a specific test file during development:

```bash
pytest -sv tests/test_specific_feature.py
```

## Submitting Issues & Pull Requests

Use the templates for required fields and examples.

- **Issues:** Follow the [ticket template](./.github/ISSUE_TEMPLATE/bug-report.yml).
- **Pull requests:** Rebase on `upstream/main`, use a descriptive branch (don't work on `main`), run `pre-commit` and tests locally, and follow the [PR template](./.github/PULL_REQUEST_TEMPLATE.md).

One member of the LeRobot team will then review your contribution.

Thank you for contributing to LeRobot!
