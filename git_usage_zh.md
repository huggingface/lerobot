# Git 使用文档

本文档面向参与本工程的协作者。默认大家的本地目录结构与本工程一致，都是基于 LeRobot 仓库展开；区别是：部分协作者的工程中可能还没有 `kinematics_lab.py`、`kinematics_lab_zh.md`，以及与其配套的 SO101 URDF/STL 资源。

## 推荐仓库关系

建议保留两个远程仓库：

```text
upstream    官方 LeRobot 仓库
origin      自己或团队维护的工程仓库
```

查看当前远程仓库：

```powershell
git remote -v
```

如果当前 `origin` 仍然指向官方 LeRobot：

```text
origin  https://github.com/huggingface/lerobot.git
```

建议改名为 `upstream`：

```powershell
git remote rename origin upstream
```

然后添加团队自己的仓库为 `origin`：

```powershell
git remote add origin <你的团队仓库地址>
```

例如：

```powershell
git remote add origin https://github.com/<user-or-org>/<repo-name>.git
```

再次确认：

```powershell
git remote -v
```

期望看到类似结果：

```text
origin    https://github.com/<user-or-org>/<repo-name>.git (fetch)
origin    https://github.com/<user-or-org>/<repo-name>.git (push)
upstream  https://github.com/huggingface/lerobot.git (fetch)
upstream  https://github.com/huggingface/lerobot.git (push)
```

## 第一次获取团队工程

如果还没有本地仓库，直接克隆团队仓库：

```powershell
git clone <你的团队仓库地址>
cd lerobot
```

如果已经有一个基于官方 LeRobot 的本地目录，可以在该目录里添加团队仓库：

```powershell
git remote rename origin upstream
git remote add origin <你的团队仓库地址>
git fetch origin
```

然后切到团队分支或同步团队主分支：

```powershell
git switch main
git pull origin main
```

如果本地有未提交改动，`git pull` 可能会失败。请先查看状态：

```powershell
git status
```

如果改动需要保留，先提交或临时保存：

```powershell
git add <文件路径>
git commit -m "Describe local changes"
```

或者：

```powershell
git stash push -m "local work before pulling team changes"
git pull origin main
git stash pop
```

## Git LFS

本工程中的 STL、视频、模型权重等大文件通过 Git LFS 管理。仓库的 `.gitattributes` 已包含：

```text
*.stl filter=lfs diff=lfs merge=lfs -text
```

第一次使用前请安装并初始化 Git LFS：

```powershell
git lfs install
```

拉取仓库后，如果发现 URDF 旁边的 `.stl` 文件很小，内容类似 LFS 指针，而不是实际模型文件，请执行：

```powershell
git lfs pull
```

查看 LFS 状态：

```powershell
git lfs status
```

## 换行符建议

Windows 用户建议在本仓库中关闭自动换行转换，避免出现大量无意义 diff：

```powershell
git config core.autocrlf false
```

查看当前配置：

```powershell
git config --get core.autocrlf
```

期望输出：

```text
false
```

## 获取 kinematics_lab 和 URDF 资源

如果你的本地工程中还没有运动学实验脚本和配套资源，拉取团队仓库后应能获得以下文件：

```text
src/lerobot/scripts/kinematics_lab.py
src/lerobot/scripts/kinematics_lab_zh.md
assets/urdf/so101_new_calib.urdf
assets/urdf/assets/*.stl
```

同步方式：

```powershell
git fetch origin
git pull origin main
git lfs pull
```

如果团队把运动学内容放在单独分支，例如 `kinematics-lab`，可以这样获取：

```powershell
git fetch origin
git switch kinematics-lab
git lfs pull
```

或者只从该分支取出相关文件到当前分支：

```powershell
git fetch origin
git checkout origin/kinematics-lab -- src/lerobot/scripts/kinematics_lab.py
git checkout origin/kinematics-lab -- src/lerobot/scripts/kinematics_lab_zh.md
git checkout origin/kinematics-lab -- assets/urdf
git lfs pull
```

取出后提交到自己的当前分支：

```powershell
git add src/lerobot/scripts/kinematics_lab.py
git add src/lerobot/scripts/kinematics_lab_zh.md
git add assets/urdf
git commit -m "Add SO101 kinematics lab and URDF assets"
```

## 日常开发流程

开始工作前先同步团队仓库：

```powershell
git switch main
git pull origin main
```

为自己的任务创建分支：

```powershell
git switch -c <branch-name>
```

例如：

```powershell
git switch -c docs/update-kinematics-guide
```

开发过程中随时查看状态：

```powershell
git status
```

查看具体修改：

```powershell
git diff
```

暂存文件：

```powershell
git add <文件路径>
```

提交：

```powershell
git commit -m "简短说明本次改动"
```

推送到团队仓库：

```powershell
git push -u origin <branch-name>
```

如果直接在 `main` 上工作：

```powershell
git push origin main
```

更推荐使用任务分支，通过 Pull Request 或 Merge Request 合并到 `main`。

## 从官方 LeRobot 同步更新

如果需要把官方 LeRobot 的最新改动合入团队工程：

```powershell
git fetch upstream
git switch main
git pull origin main
git merge upstream/main
```

如果有冲突，Git 会提示冲突文件。解决冲突后：

```powershell
git status
git add <已解决的文件>
git commit
git push origin main
```

也可以在单独分支中同步官方更新：

```powershell
git switch -c chore/sync-upstream
git merge upstream/main
git push -u origin chore/sync-upstream
```

这样更方便团队审核。

## 不建议提交的内容

以下内容通常不应该提交：

```text
__pycache__/
*.pyc
src/lerobot.egg-info/
.venv/
venv/
data/
outputs/
external/
logs/
tmp/
```

其中 `external/` 通常用于临时克隆第三方仓库，例如 SO-ARM100 原始资源。真正需要版本管理的 URDF/STL，应放在本工程的 `assets/urdf/` 下，并通过 Git LFS 管理。

提交前可以检查是否有不该提交的文件：

```powershell
git status --ignored --short
```

## 推荐提交粒度

建议按功能拆分提交，而不是把所有改动一次性提交。

示例：

```powershell
git add src/lerobot/scripts/kinematics_lab.py
git commit -m "Add interactive kinematics lab"

git add src/lerobot/scripts/kinematics_lab_zh.md
git commit -m "Add Chinese kinematics lab guide"

git add assets/urdf
git commit -m "Add SO101 URDF assets"
```

如果某次改动同时包含代码、文档和资源，也可以合并成一个清晰的提交：

```powershell
git add src/lerobot/scripts/kinematics_lab.py
git add src/lerobot/scripts/kinematics_lab_zh.md
git add assets/urdf
git commit -m "Add SO101 kinematics lab and assets"
```

## 常见问题

### pull 时提示本地文件会被覆盖

说明本地有未提交改动。先查看：

```powershell
git status
```

如果要保留这些改动：

```powershell
git add <文件路径>
git commit -m "Save local changes"
git pull origin main
```

如果只是临时改动：

```powershell
git stash push -m "temporary local changes"
git pull origin main
git stash pop
```

### STL 文件拉下来无法正常使用

先确认 Git LFS 已安装：

```powershell
git lfs version
```

然后拉取 LFS 文件：

```powershell
git lfs pull
```

### 不小心提交了不该提交的缓存文件

如果还没有 push，可以从暂存区移除：

```powershell
git restore --staged <文件路径>
```

如果已经提交但还没 push，可以新提交一次删除：

```powershell
git rm -r <文件路径>
git commit -m "Remove generated files"
```

不要随意使用 `git reset --hard`，它会丢弃本地未提交改动。

### 想确认某个文件来自哪个分支

查看分支列表：

```powershell
git branch -a
```

查看某个分支中是否有文件：

```powershell
git ls-tree -r origin/main -- src/lerobot/scripts/kinematics_lab.py
```

查看文件历史：

```powershell
git log -- src/lerobot/scripts/kinematics_lab.py
```

## 建议的协作规则

- `main` 保持可运行、可拉取。
- 大功能在独立分支完成，再合并到 `main`。
- 提交信息用一句话说明“做了什么”，例如 `Add SO101 kinematics guide`。
- 提交前运行必要测试或至少执行相关脚本的干跑检查。
- URDF/STL 等资源放在 `assets/`，临时外部仓库放在 `external/`。
- 不提交虚拟环境、缓存、运行输出和个人 IDE 配置。
