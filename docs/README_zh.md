英文版本请见：[`README.md`](./README.md)

# MagiAttention 文档贡献指南

本指南详细说明如何构建、预览和贡献 MagiAttention 文档。无需任何 Sphinx 经验即可上手。

## 前置条件

- Python 3.8+
- `pip`（Python 自带）
- 终端（bash、zsh、PowerShell 等均可）
- 文本编辑器（VS Code、Vim 等均可）

## 快速上手（5 分钟）

```bash
# 1. 进入 docs 目录
cd docs

# 2. 安装依赖（仅首次需要）
pip install -r requirements.txt

# 3. 构建文档
make html

# 4. 在浏览器中打开
open build/html/index.html        # macOS
xdg-open build/html/index.html    # Linux
start build/html/index.html       # Windows (Git Bash)
```

完成！你应该能在浏览器中看到文档了。

> **提示：** 如果看到旧内容，运行 `make clean && make html` 强制全量重建。

---

## 目录结构

```
docs/
├── source/                  # 所有文档源文件都在这里
│   ├── index.md             # 首页
│   ├── conf.py              # Sphinx 配置文件（一般不需要改）
│   ├── _static/             # 自定义 CSS、文档引用的图片
│   ├── _templates/          # HTML 模板（如语言切换器）
│   ├── user_guide/          # 面向用户的指南
│   │   ├── toc.md           # 用户指南章节的目录
│   │   ├── install.md       # 安装指南
│   │   ├── quickstart.md    # 快速开始
│   │   ├── magi_api.md      # API 参考
│   │   └── env_variables.md # 环境变量参考
│   └── blog/                # 技术博客文章
│       ├── toc.md           # 博客章节的目录
│       ├── refs/            # BibTeX 引用文件（每篇博客一个）
│       │   └── *.bib
│       └── *.md             # 博客文章
├── locale/                  # 中文翻译文件（.po）
│   └── zh_CN/LC_MESSAGES/
│       ├── index.po
│       ├── user_guide/*.po
│       └── blog/*.po
├── build/                   # 生成的输出（已被 git 忽略）
├── requirements.txt         # Python 依赖
├── Makefile                 # 构建命令（Linux/macOS）
└── make.bat                 # 构建命令（Windows）
```

---

## 如何编写文档

所有文档都使用 **Markdown** 编写，采用 [MyST 语法](https://myst-parser.readthedocs.io/en/latest/)——它在标准 Markdown 基础上扩展了 Sphinx 特有的功能。

### 新增一篇用户指南

1. 在 `source/user_guide/` 下创建一个新的 `.md` 文件：

```markdown
# 我的新页面标题

## 第一节

这里写内容。可以使用 **粗体**、*斜体*、`行内代码` 等。

### 子节

更多内容...

## 第二节

另一节内容...
```

2. 在 `source/user_guide/toc.md` 中注册——只需添加文件名（不含 `.md` 后缀也行，带 `.md` 也行）：

```markdown
# User Guide

\`\`\`{toctree}
:caption: User Guide
:maxdepth: 2

install.md
quickstart.md
magi_api.md
env_variables.md
my_new_page.md          # <-- 在这里添加你的新文件
\`\`\`
```

3. 构建并预览：

```bash
make clean && make html
open build/html/index.html
```

### 新增一篇博客文章

博客文章需要一个 YAML frontmatter 头部来声明元数据：

1. 创建 `source/blog/my_topic.md`：

```markdown
---
blogpost: true
date: Mar 31, 2026
author: 你的名字
location: China
category: MagiAttention
tags: 标签1, 标签2, 标签3
language: English
---

# 我的博客文章标题

## Introduction

你的内容...

## Citation

If you find MagiAttention useful in your research, please cite:

\`\`\`bibtex
@misc{magiattention2025,
  title={MagiAttention: ...},
  author={...},
  year={2025},
  howpublished={\url{https://github.com/SandAI-org/MagiAttention/}},
}
\`\`\`

## References

\`\`\`{bibliography}
:filter: docname in docnames
\`\`\`
```

2. 如果博客文章中有引用文献，创建 `source/blog/refs/my_topic.bib`，并在 `source/conf.py` 的 `blog_titles` 列表中添加博客标题：

```python
blog_titles = [
    "magi_attn",
    "cp_benchmark",
    # ... 已有条目 ...
    "my_topic",       # <-- 在这里添加
]
```

3. 在 `source/blog/toc.md` 中注册：

```markdown
\`\`\`{toctree}
:caption: Blogs
:maxdepth: 1

magi_attn.md
# ... 已有条目 ...
my_topic.md             # <-- 在这里添加
\`\`\`
```

4. 构建并预览。

---

## MyST Markdown 速查表

### 基础格式

| 语法 | 效果 |
|------|------|
| `**粗体**` | **粗体** |
| `*斜体*` | *斜体* |
| `` `代码` `` | `代码` |
| `[文字](链接)` | 超链接 |
| `![替代文字](路径)` | 图片 |

### 代码块

````markdown
```python
def hello():
    print("Hello, world!")
```
````

### 数学公式（LaTeX）

行内公式：`` {math}`E = mc^2` ``

块级公式：

```markdown
$$
\frac{\partial f}{\partial x} = 2x
$$
```

### 提示框（Note、Warning、Tip）

```markdown
:::{note}
这是一个提示框。
:::

:::{warning}
这是一个警告框。
:::

:::{tip}
这是一个建议框。
:::
```

### 带标题的图片

```markdown
:::{figure} ../../assets/path/to/image.png
:name: my_figure_label
:width: 80%

这是图片说明文字。
:::
```

在其他地方引用它：`` {numref}`my_figure_label` ``

### 页内目录

```markdown
\`\`\`{contents}
:local: true
\`\`\`
```

这会根据当前页面的标题自动生成一个小型目录。

### 跨页面引用

```markdown
详见 [安装指南](../user_guide/install.md)。
```

### 引用文献（BibTeX）

在 `.bib` 文件中：
```bibtex
@article{dao2022flashattention,
  title={FlashAttention},
  author={Dao, Tri},
  year={2022}
}
```

在 `.md` 文件中：
```markdown
如 {cite}`dao2022flashattention` 所示，...
```

更完整的 MyST 语法请参考官方文档：[MyST documentation](https://myst-parser.readthedocs.io/en/latest/)。

主题特有的组件（标签页、网格、卡片、徽章等）请参考 [PyData Theme User Guide](https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/index.html)。

---

## 中英文双语文档

所有文档以**英文为主源**，中文翻译通过 `.po` 文件管理，遵循标准的 Sphinx 国际化（i18n）工作流。

### 工作原理

```
source/*.md（英文源文件，"唯一真相源"）
     │
     ▼  make update-po
locale/zh_CN/LC_MESSAGES/*.po（中文翻译）
     │
     ▼  make html-multilang
build/html/en/   （英文站点）
build/html/zh_CN/ （中文站点，带语言切换器）
```

### 完整工作流

#### 第一步：编写或修改英文源文件

照常编辑 `source/` 下的 `.md` 文件。

#### 第二步：提取翻译字符串

```bash
make update-po
```

该命令会扫描所有 `.md` 文件，在 `locale/zh_CN/LC_MESSAGES/` 下创建或更新 `.po` 文件。每个 `.po` 文件对应一个 `.md` 源文件。

#### 第三步：翻译 `.po` 文件

打开对应的 `.po` 文件（例如 `locale/zh_CN/LC_MESSAGES/user_guide/install.po`），格式如下：

```po
#: ../../source/user_guide/install.md:1
msgid "Installation"
msgstr "安装"

#: ../../source/user_guide/install.md:14
msgid "Activate an NGC-PyTorch Container"
msgstr "启动 NGC-PyTorch 容器"
```

- `msgid` = 英文原文（**不要修改**）
- `msgstr` = 中文翻译（**填写这里**）

**翻译规则：**
- 所有技术术语保留英文（FFA、MagiAttention、CUDA、NVLink 等）
- 所有 LaTeX 公式、代码块、URL、引用文献和格式标记保持原样
- 只翻译自然语言描述
- 如果 `msgstr` 留空 `""`，该条目将回退显示英文

#### 第四步：构建双语文档

```bash
make html-multilang
```

输出：
- 英文站点：`build/html/en/index.html`
- 中文站点：`build/html/zh_CN/index.html`

网页顶部导航栏包含语言切换器，可在 `English` 和 `简体中文` 之间跳转。

#### 第五步：预览

```bash
# 英文
open build/html/en/index.html

# 中文
open build/html/zh_CN/index.html
```

### 新增页面后的翻译流程

添加新 `.md` 文件并注册到 toctree 后：

```bash
make update-po          # 为新页面生成对应的 .po 文件
# ... 翻译新的 .po 文件 ...
make html-multilang     # 构建双语文档
```

---

## 可用的 Make 命令

| 命令 | 说明 |
|------|------|
| `make html` | 构建英文文档（默认，输出到 `build/html/`） |
| `make clean` | 清除所有构建产物 |
| `make update-po` | 提取字符串并更新中文 `.po` 文件 |
| `make html-en` | 仅构建英文（输出到 `build/html/en/`） |
| `make html-zh` | 仅构建中文（输出到 `build/html/zh_CN/`） |
| `make html-multilang` | 同时构建英文和中文 |

---

## 常见问题

### "sphinx-build: command not found"

需要先安装依赖：

```bash
pip install -r requirements.txt
```

### 构建时出现交叉引用警告

类似 `myst.xref_missing` 的警告通常无害——它们表示 Sphinx 无法静态解析的锚点链接，但这些链接在浏览器中仍然可以正常工作。

### 修改后内容没有更新

执行全量重建：

```bash
make clean && make html
```

### 中文站点显示英文

对应页面的 `.po` 文件中有空的 `msgstr` 条目。填入翻译后重新构建即可。
