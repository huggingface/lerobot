# How to contribute to 🤗 LeRobot?

Everyone is welcome to contribute, and we value everybody's contribution. Code
is thus not the only way to help the community. Answering questions, helping
others, reaching out and improving the documentations are immensely valuable to
the community.

It also helps us if you spread the word: reference the library from blog posts
on the awesome projects it made possible, shout out on Twitter when it has
helped you, or simply ⭐️ the repo to say "thank you".

Whichever way you choose to contribute, please be mindful to respect our
[code of conduct](https://github.com/huggingface/lerobot/blob/main/CODE_OF_CONDUCT.md).

## You can contribute in so many ways!

Some of the ways you can contribute to 🤗 LeRobot:
* Fixing outstanding issues with the existing code.
* Implementing new models, datasets or simulation environments.
* Contributing to the examples or to the documentation.
* Submitting issues related to bugs or desired new features.

Following the guides below, feel free to open issues and PRs and to coordinate your efforts with the community on our [Discord Channel](https://discord.gg/VjFz58wn3R). For specific inquiries, reach out to [Remi Cadene](mailto:remi.cadene@huggingface.co).

If you are not sure how to contribute or want to know the next features we working on, look on this project page: [LeRobot TODO](https://github.com/orgs/huggingface/projects/46)

## Submitting a new issue or feature request

Do your best to follow these guidelines when submitting an issue or a feature
request. It will make it easier for us to come back to you quickly and with good
feedback.

### Did you find a bug?

The 🤗 LeRobot library is robust and reliable thanks to the users who notify us of
the problems they encounter. So thank you for reporting an issue.

First, we would really appreciate it if you could **make sure the bug was not
already reported** (use the search bar on Github under Issues).

Did not find it? :( So we can act quickly on it, please follow these steps:

* Include your **OS type and version**, the versions of **Python** and **PyTorch**.
* A short, self-contained, code snippet that allows us to reproduce the bug in
  less than 30s.
* The full traceback if an exception is raised.
* Attach any other additional information, like screenshots, you think may help.

### Do you want a new feature?

A good feature request addresses the following points:

1. Motivation first:
* Is it related to a problem/frustration with the library? If so, please explain
  why. Providing a code snippet that demonstrates the problem is best.
* Is it related to something you would need for a project? We'd love to hear
  about it!
* Is it something you worked on and think could benefit the community?
  Awesome! Tell us what problem it solved for you.
2. Write a *paragraph* describing the feature.
3. Provide a **code snippet** that demonstrates its future use.
4. In case this is related to a paper, please attach a link.
5. Attach any additional information (drawings, screenshots, etc.) you think may help.

If your issue is well written we're already 80% of the way there by the time you
post it.

## Adding new policies, datasets or environments

Look at our implementations for [datasets](./src/lerobot/datasets/), [policies](./src/lerobot/policies/),
environments ([aloha](https://github.com/huggingface/gym-aloha),
[xarm](https://github.com/huggingface/gym-xarm),
[pusht](https://github.com/huggingface/gym-pusht))
and follow the same api design.

When implementing a new dataset loadable with LeRobotDataset follow these steps:
- Update `available_datasets_per_env` in `lerobot/__init__.py`

When implementing a new environment (e.g. `gym_aloha`), follow these steps:
- Update `available_tasks_per_env` and `available_datasets_per_env` in `lerobot/__init__.py`

When implementing a new policy class (e.g. `DiffusionPolicy`) follow these steps:
- Update `available_policies` and `available_policies_per_env`, in `lerobot/__init__.py`
- Set the required `name` class attribute.
- Update variables in `tests/test_available.py` by importing your new Policy class

## Submitting a pull request (PR)

Before writing code, we strongly advise you to search through the existing PRs or
issues to make sure that nobody is already working on the same thing. If you are
unsure, it is always a good idea to open an issue to get some feedback.

You will need basic `git` proficiency to be able to contribute to
🤗 LeRobot. `git` is not the easiest tool to use but it has the greatest
manual. Type `git --help` in a shell and enjoy. If you prefer books, [Pro
Git](https://git-scm.com/book/en/v2) is a very good reference.

Follow these steps to start contributing:

1. Fork the [repository](https://github.com/huggingface/lerobot) by
   clicking on the 'Fork' button on the repository's page. This creates a copy of the code
   under your GitHub user account.

2. Clone your fork to your local disk, and add the base repository as a remote. The following command
   assumes you have your public SSH key uploaded to GitHub. See the following guide for more
   [information](https://docs.github.com/en/repositories/creating-and-managing-repositories/cloning-a-repository).

   ```bash
   git clone git@github.com:<your Github handle>/lerobot.git
   cd lerobot
   git remote add upstream https://github.com/huggingface/lerobot.git
   ```

3. Create a new branch to hold your development changes, and do this for every new PR you work on.

   Start by synchronizing your `main` branch with the `upstream/main` branch (more details in the [GitHub Docs](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork)):

   ```bash
   git checkout main
   git fetch upstream
   git rebase upstream/main
   ```

   Once your `main` branch is synchronized, create a new branch from it:

   ```bash
   git checkout -b a-descriptive-name-for-my-changes
   ```

   🚨 **Do not** work on the `main` branch.

4. for development, we advise to use a tool like `poetry` or `uv` instead of just `pip` to easily track our dependencies.
   Follow the instructions to [install poetry](https://python-poetry.org/docs/#installation) (use a version >=2.1.0) or to [install uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods) if you don't have one of them already.

   Set up a development environment with conda or miniconda:
   ```bash
   conda create -y -n lerobot-dev python=3.10 && conda activate lerobot-dev
   ```

   If you're using `uv`, it can manage python versions so you can instead do:
   ```bash
   uv venv --python 3.10 && source .venv/bin/activate
   ```

   To develop on 🤗 LeRobot, you will at least need to install the `dev` and `test` extras dependencies along with the core library:

   using `poetry`
   ```bash
   poetry sync --extras "dev test"
   ```

   using `uv`
   ```bash
   uv sync --extra dev --extra test
   ```

   You can also install the project with all its dependencies (including environments):

   using `poetry`
   ```bash
   poetry sync --all-extras
   ```

   using `uv`
   ```bash
   uv sync --all-extras
   ```

   > **Note:** If you don't install simulation environments with `--all-extras`, the tests that require them will be skipped when running the pytest suite locally. However, they *will* be tested in the CI. In general, we advise you to install everything and test locally before pushing.

   Whichever command you chose to install the project (e.g. `poetry sync --all-extras`), you should run it again when pulling code with an updated version of `pyproject.toml` and `poetry.lock` in order to synchronize your virtual environment with the new dependencies.

   The equivalent of `pip install some-package`, would just be:

   using `poetry`
   ```bash
   poetry add some-package
   ```

   using `uv`
   ```bash
   uv add some-package
   ```

   When making changes to the poetry sections of the `pyproject.toml`, you should run the following command to lock dependencies.
   using `poetry`
   ```bash
   poetry lock
   ```

   using `uv`
   ```bash
   uv lock
   ```


5. Develop the features on your branch.

   As you work on the features, you should make sure that the test suite
   passes. You should run the tests impacted by your changes like this (see
   below an explanation regarding the environment variable):

   ```bash
   pytest tests/<TEST_TO_RUN>.py
   ```

6. Follow our style.

   `lerobot` relies on `ruff` to format its source code
   consistently. Set up [`pre-commit`](https://pre-commit.com/) to run these checks
   automatically as Git commit hooks.

   Install `pre-commit` hooks:
   ```bash
   pre-commit install
   ```

   You can run these hooks whenever you need on staged files with:
   ```bash
   pre-commit
   ```

   Once you're happy with your changes, add changed files using `git add` and
   make a commit with `git commit` to record your changes locally:

   ```bash
   git add modified_file.py
   git commit
   ```

   Note, if you already committed some changes that have a wrong formatting, you can use:
   ```bash
   pre-commit run --all-files
   ```

   Please write [good commit messages](https://chris.beams.io/posts/git-commit/).

   It is a good idea to sync your copy of the code with the original
   repository regularly. This way you can quickly account for changes:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Push the changes to your account using:

   ```bash
   git push -u origin a-descriptive-name-for-my-changes
   ```

6. Once you are satisfied (**and the checklist below is happy too**), go to the
   webpage of your fork on GitHub. Click on 'Pull request' to send your changes
   to the project maintainers for review.

7. It's ok if maintainers ask you for changes. It happens to core contributors
   too! So everyone can see the changes in the Pull request, work in your local
   branch and push the changes to your fork. They will automatically appear in
   the pull request.


### Checklist

1. The title of your pull request should be a summary of its contribution;
2. If your pull request addresses an issue, please mention the issue number in
   the pull request description to make sure they are linked (and people
   consulting the issue know you are working on it);
3. To indicate a work in progress please prefix the title with `[WIP]`, or preferably mark
   the PR as a draft PR. These are useful to avoid duplicated work, and to differentiate
   it from PRs ready to be merged;
4. Make sure existing tests pass;

### Tests

An extensive test suite is included to test the library behavior and several examples. Library tests can be found in the [tests folder](https://github.com/huggingface/lerobot/tree/main/tests).

Install [git lfs](https://git-lfs.com/) to retrieve test artifacts (if you don't have it already).

On Mac:
```bash
brew install git-lfs
git lfs install
```

On Ubuntu:
```bash
sudo apt-get install git-lfs
git lfs install
```

Pull artifacts if they're not in [tests/artifacts](tests/artifacts)
```bash
git lfs pull
```

We use `pytest` in order to run the tests. From the root of the
repository, here's how to run tests with `pytest` for the library:

```bash
python -m pytest -sv ./tests
```


You can specify a smaller set of tests in order to test only the feature
you're working on.
