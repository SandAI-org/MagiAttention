# Deploy a New Documentation Version

This guide explains how to publish a new version of the MagiAttention docs to GitHub Pages (e.g. `v1.2.0`).

**Prerequisites:** You need write access to the repo. The workflow in `.github/workflows/sphinx.yaml` builds and deploys automatically on push.

---

## Overview

| Step | What you do |
|------|-------------|
| 1 | Register the version on `gh-pages` |
| 2 | Create a temporary deploy branch from your source branch |
| 3 | Point Sphinx + CI at the new version |
| 4 | Push → CI deploys → delete the deploy branch |
| 5 | Update default redirects on `gh-pages` |

---

## Step 1: Register the version on `gh-pages`

```bash
git checkout gh-pages
```

Edit `docs/versions.json` and add a new entry (usually at the top, after `main`):

```json
{
    "name": "v1.2.0",
    "version": "v1.2.0",
    "url": "https://sandai-org.github.io/MagiAttention/docs/v1.2.0/"
}
```

| Field | Meaning |
|-------|---------|
| `name` | Label shown in the version dropdown |
| `version` | Unique key; must match `version_match` in `conf.py` |
| `url` | Public URL after deploy; path must match `target-folder` below |

Commit and push `gh-pages`, then switch back to your working branch:

```bash
git add docs/versions.json
git commit -m "docs: add v1.2.0 to version switcher"
git push origin gh-pages
git checkout main   # or your feature branch
```

---

## Step 2: Create a deploy branch

From the branch whose docs you want to publish (e.g. `main` or a release tag):

```bash
git checkout -b deploy_v1.2.0
```

---

## Step 3: Configure Sphinx and CI

### 3.1 `docs/source/conf.py`

Set the release string (around line 31):

```python
release = "v1.2.0"
```

Set the version switcher key (in `html_theme_options` → `switcher`):

```python
"version_match": "v1.2.0",   # must match "version" in versions.json
```

### 3.2 `.github/workflows/sphinx.yaml`

Trigger the workflow only from your deploy branch:

```yaml
on:
  push:
    branches:
      - deploy_v1.2.0
```

Set the deploy target folder (must match the path in `url`, without trailing slash):

```yaml
target-folder: docs/v1.2.0
```

---

## Step 4: Deploy

```bash
git add docs/source/conf.py .github/workflows/sphinx.yaml
git commit -m "docs: deploy v1.2.0"
git push origin deploy_v1.2.0
```

GitHub Actions will build the docs and push them to `gh-pages` under `docs/v1.2.0/`.

1. Open **Actions** → **sphinx deploy** and wait for a green run.
2. Open `https://sandai-org.github.io/MagiAttention/docs/v1.2.0/` and spot-check the site.
3. Delete the deploy branch (local and remote):

---

## Step 5: Set the default version (optional)

If this version should be the **default** when users visit the docs root:

```bash
git checkout gh-pages
```

Update both redirect files to point at the new version:

**`index.html`** (repo root):

```html
<meta http-equiv="refresh" content="0; url=./docs/v1.2.0" />
```

**`docs/index.html`**:

```html
<meta http-equiv="refresh" content="0; url=./v1.2.0" />
```

Commit, push `gh-pages`, then return to your working branch.

---
