<!---
Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# 生成文档

要生成文档，首先需要构建它。构建文档需要几个包，你可以在代码仓库的根目录下使用以下命令安装它们：

```bash
pip install -e . -r docs-requirements.txt
```

你还需要 `nodejs`。请参考他们的 [安装页面](https://nodejs.org/en/download)

---

**注意**

你只需要在本地生成文档来检查它（例如，如果你计划进行更改并想在提交之前检查它们的样子）。你不需要 `git commit` 构建的文档。

---

## 构建文档

设置好 `doc-builder` 和其他包后，你可以通过输入以下命令来生成文档：

```bash
doc-builder build lerobot docs/source/ --build_dir ~/tmp/test-build
```

你可以调整 `--build_dir` 来设置任何你喜欢的临时文件夹。这个命令会创建它并生成 MDX 文件，这些文件将在主网站上渲染为文档。你可以在你喜欢的 Markdown 编辑器中检查它们。

## 预览文档

要预览文档，首先使用以下命令安装 `watchdog` 模块：

```bash
pip install watchdog
```

然后运行以下命令：

```bash
doc-builder preview lerobot docs/source/
```

文档可以在 [http://localhost:3000](http://localhost:3000) 查看。你也可以在打开 PR 后预览文档。你会看到一个机器人添加评论到一个链接，其中包含你的更改的文档。

---

**注意**

`preview` 命令只适用于现有的文档文件。当你添加一个全新的文件时，你需要更新 `_toctree.yml` 并重新启动 `preview` 命令（按 `ctrl-c` 停止它并再次调用 `doc-builder preview ...`）。

---

## 向导航栏添加新元素

接受的文件是 Markdown (.md)。

创建一个文件及其扩展名，并将其放在 source 目录中。然后你可以通过在 [`_toctree.yml`](https://github.com/huggingface/lerobot/blob/main/docs/source/_toctree.yml) 文件中放置不带扩展名的文件名来将该文件链接到 toc-tree。

## 重命名节标题和移动节

当重命名节标题和/或将节从一个文档移动到另一个文档时，保持旧链接有效会有所帮助。这是因为旧链接可能会在 Issues、论坛和社交媒体中使用，如果几个月后阅读这些内容的用户仍然可以轻松导航到最初的信息，那将提供更好的用户体验。

因此，我们只是在原始节所在的文档末尾保留一个移动节的小地图。关键是保留原始锚点。

所以如果你将节从 "Section A" 重命名为 "Section B"，那么你可以在文件末尾添加：

```
被移动的节：

[ <a href="#section-b">Section A</a><a id="section-a"></a> ]
```

当然，如果你将它移动到另一个文件，那么：

```
被移动的节：

[ <a href="../new-file#section-b">Section A</a><a id="section-a"></a> ]
```

使用相对样式链接到新文件，以便版本化文档继续工作。

有关丰富的移动节集的示例，请参阅 [transformers Trainer 文档](https://github.com/huggingface/transformers/blob/main/docs/source/en/main_classes/trainer.md) 的末尾。

### 添加新教程

添加新教程或节分两步完成：

- 在 `./source` 下添加新文件。该文件可以是 ReStructuredText (.rst) 或 Markdown (.md)。
- 在 `./source/_toctree.yml` 中的正确 toc-tree 上链接该文件。

确保将新文件放在适当的部分。如有疑问，可以在 Github Issue 或 PR 中提问。

### 编写源文档

应该放在 `code` 中的值应该用反引号包围：\`像这样\`。请注意，参数名称和对象如 True、None 或任何字符串通常应该放在 `code` 中。

#### 编写多行代码块

多行代码块对于显示示例很有用。它们像 Markdown 中通常一样在两行三个反引号之间完成：

````
```
# 第一行代码
# 第二行
# 等等
```
````

#### 添加图片

由于仓库快速增长，重要的是确保不添加会显著增加仓库重量的文件。这包括图片、视频和其他非文本文件。我们倾向于利用 hf.co 托管的 `dataset`，如 [`hf-internal-testing`](https://huggingface.co/hf-internal-testing) 上托管的那些，来放置这些文件并通过 URL 引用它们。我们建议将它们放在以下数据集中：[huggingface/documentation-images](https://huggingface.co/datasets/huggingface/documentation-images)。如果是外部贡献，请随意将图片添加到你的 PR 中，并请 Hugging Face 成员将你的图片迁移到这个数据集。
