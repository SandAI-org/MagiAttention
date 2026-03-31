Chinese version: [`README_zh.md`](./README_zh.md)

# MagiAttention Documentation Guide

This guide explains how to build, preview, and contribute to the MagiAttention documentation. No prior Sphinx experience is required.

## Prerequisites

- Python 3.8+
- `pip` (comes with Python)
- A terminal (bash, zsh, PowerShell, etc.)
- A text editor (VS Code, Vim, etc.)

## Quick Start (5 minutes)

```bash
# 1. Enter the docs directory
cd docs

# 2. Install dependencies (one-time)
pip install -r requirements.txt

# 3. Build the docs
make html

# 4. Open in browser
open build/html/index.html        # macOS
xdg-open build/html/index.html    # Linux
start build/html/index.html       # Windows (Git Bash)
```

That's it! You should see the documentation in your browser.

> **Tip:** Run `make clean && make html` to force a full rebuild if you see stale content.

---

## Directory Structure

```
docs/
├── source/                  # All documentation source files live here
│   ├── index.md             # Homepage
│   ├── conf.py              # Sphinx configuration (rarely need to edit)
│   ├── _static/             # Custom CSS, images referenced by docs
│   ├── _templates/          # HTML templates (e.g. language switcher)
│   ├── user_guide/          # User-facing guides
│   │   ├── toc.md           # Table of contents for user guide section
│   │   ├── install.md       # Installation guide
│   │   ├── quickstart.md    # Quick start guide
│   │   ├── magi_api.md      # API reference
│   │   └── env_variables.md # Environment variables reference
│   └── blog/                # Technical blog posts
│       ├── toc.md           # Table of contents for blog section
│       ├── refs/            # BibTeX citation files (one per blog post)
│       │   └── *.bib
│       └── *.md             # Blog post files
├── locale/                  # Chinese translation files (.po)
│   └── zh_CN/LC_MESSAGES/
│       ├── index.po
│       ├── user_guide/*.po
│       └── blog/*.po
├── build/                   # Generated output (git-ignored)
├── requirements.txt         # Python dependencies
├── Makefile                 # Build commands (Linux/macOS)
└── make.bat                 # Build commands (Windows)
```

---

## How to Write Documentation

All documentation is written in **Markdown** using [MyST syntax](https://myst-parser.readthedocs.io/en/latest/), which extends standard Markdown with Sphinx-specific features.

### Adding a New User Guide Page

1. Create a new `.md` file in `source/user_guide/`:

```markdown
# My New Page Title

## Section One

Some content here. You can use **bold**, *italic*, `inline code`, etc.

### Subsection

More content...

## Section Two

Another section...
```

2. Register it in `source/user_guide/toc.md` — just add the filename (without `.md`):

```markdown
# User Guide

\`\`\`{toctree}
:caption: User Guide
:maxdepth: 2

install.md
quickstart.md
magi_api.md
env_variables.md
my_new_page.md          # <-- add your new file here
\`\`\`
```

3. Build and preview:

```bash
make clean && make html
open build/html/index.html
```

### Adding a New Blog Post

Blog posts have a YAML frontmatter header for metadata:

1. Create `source/blog/my_topic.md`:

```markdown
---
blogpost: true
date: Mar 31, 2026
author: Your Name
location: China
category: MagiAttention
tags: Tag1, Tag2, Tag3
language: English
---

# My Blog Post Title

## Introduction

Your content here...

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

2. If your blog post has citations, create `source/blog/refs/my_topic.bib` and add the blog title to the `blog_titles` list in `source/conf.py`:

```python
blog_titles = [
    "magi_attn",
    "cp_benchmark",
    # ... existing entries ...
    "my_topic",       # <-- add here
]
```

3. Register in `source/blog/toc.md`:

```markdown
\`\`\`{toctree}
:caption: Blogs
:maxdepth: 1

magi_attn.md
# ... existing entries ...
my_topic.md             # <-- add here
\`\`\`
```

4. Build and preview.

---

## MyST Markdown Cheat Sheet

### Basic Formatting

| Syntax | Result |
|--------|--------|
| `**bold**` | **bold** |
| `*italic*` | *italic* |
| `` `code` `` | `code` |
| `[text](url)` | hyperlink |
| `![alt](path)` | image |

### Code Blocks

````markdown
```python
def hello():
    print("Hello, world!")
```
````

### Math (LaTeX)

Inline: `` {math}`E = mc^2` ``

Block:

```markdown
$$
\frac{\partial f}{\partial x} = 2x
$$
```

### Admonitions (Notes, Warnings, Tips)

```markdown
:::{note}
This is a note box.
:::

:::{warning}
This is a warning box.
:::

:::{tip}
This is a tip box.
:::
```

### Figures with Captions

```markdown
:::{figure} ../../assets/path/to/image.png
:name: my_figure_label
:width: 80%

This is the figure caption.
:::
```

Reference it elsewhere: `` {numref}`my_figure_label` ``

### Table of Contents in a Page

```markdown
\`\`\`{contents}
:local: true
\`\`\`
```

This generates a mini table of contents from the headings on that page.

### Cross-References to Other Pages

```markdown
See the [Installation Guide](../user_guide/install.md) for details.
```

### Citations (BibTeX)

In your `.bib` file:
```bibtex
@article{dao2022flashattention,
  title={FlashAttention},
  author={Dao, Tri},
  year={2022}
}
```

In your `.md` file:
```markdown
As shown in {cite}`dao2022flashattention`, ...
```

For a full reference of MyST syntax, see the official [MyST documentation](https://myst-parser.readthedocs.io/en/latest/).

For theme-specific components (tabs, grids, cards, badges), see the [PyData Theme User Guide](https://pydata-sphinx-theme.readthedocs.io/en/stable/user_guide/index.html).

---

## Bilingual Documentation (English + Chinese)

All documentation is maintained in English as the primary source, with Chinese translations managed via `.po` files using the standard Sphinx i18n workflow.

### How It Works

```
source/*.md  (English source, the "single source of truth")
     │
     ▼  make update-po
locale/zh_CN/LC_MESSAGES/*.po  (Chinese translations)
     │
     ▼  make html-multilang
build/html/en/   (English site)
build/html/zh_CN/ (Chinese site, with language switcher)
```

### Full Workflow

#### Step 1: Write or Edit English Source

Edit `.md` files in `source/` as usual.

#### Step 2: Extract Translation Strings

```bash
make update-po
```

This scans all `.md` files and creates/updates `.po` files under `locale/zh_CN/LC_MESSAGES/`. Each `.po` file corresponds to one `.md` source file.

#### Step 3: Translate `.po` Files

Open the `.po` file (e.g. `locale/zh_CN/LC_MESSAGES/user_guide/install.po`). It looks like this:

```po
#: ../../source/user_guide/install.md:1
msgid "Installation"
msgstr "<Chinese translation>"

#: ../../source/user_guide/install.md:14
msgid "Activate an NGC-PyTorch Container"
msgstr "<Chinese translation>"
```

- `msgid` = English original (do NOT modify)
- `msgstr` = Chinese translation (fill this in)

**Translation rules:**
- Keep all technical terms in English (FFA, MagiAttention, CUDA, NVLink, etc.)
- Keep all LaTeX math, code blocks, URLs, citations, and formatting as-is
- Only translate natural language prose
- Leave `msgstr ""` empty if you want to fall back to English

#### Step 4: Build Both Languages

```bash
make html-multilang
```

This produces:
- English site: `build/html/en/index.html`
- Chinese site: `build/html/zh_CN/index.html`

The top navigation bar includes a language switcher to jump between `English` and `Simplified Chinese`.

#### Step 5: Preview

```bash
# English
open build/html/en/index.html

# Chinese
open build/html/zh_CN/index.html
```

### When You Add a New Page

After adding a new `.md` file and registering it in the toctree:

```bash
make update-po          # generates a new .po file for the new page
# ... translate the new .po file ...
make html-multilang     # build both languages
```

---

## Available Make Commands

| Command | Description |
|---------|-------------|
| `make html` | Build English docs (default, output: `build/html/`) |
| `make clean` | Remove all build artifacts |
| `make update-po` | Extract strings and update Chinese `.po` files |
| `make html-en` | Build English only (output: `build/html/en/`) |
| `make html-zh` | Build Chinese only (output: `build/html/zh_CN/`) |
| `make html-multilang` | Build both English and Chinese |

---

## Troubleshooting

### "sphinx-build: command not found"

You need to install dependencies first:

```bash
pip install -r requirements.txt
```

### Build warnings about cross-references

Warnings like `myst.xref_missing` are usually harmless — they indicate links to anchors that Sphinx cannot resolve statically. The links still work in the browser.

### Changes not showing up

Run a clean build:

```bash
make clean && make html
```

### Chinese site shows English text

The `.po` file for that page has empty `msgstr` entries. Fill in the translations and rebuild.
