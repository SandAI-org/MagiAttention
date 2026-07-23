# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = source
BUILDDIR      = build
LOCALEDIR     = locale
GETTEXTDIR    = $(BUILDDIR)/gettext

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

.PHONY: gettext update-po html html-en html-zh html-multilang

gettext:
	@$(SPHINXBUILD) -M gettext "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Re-extract all translatable strings from the docs (via `gettext`) and merge
# them into the zh_CN `.po` catalogs under $(LOCALEDIR). New/changed source
# strings show up as untranslated or `#, fuzzy` entries that still need to be
# translated by hand.
#
# REMEMBER: run `make update-po` (and translate the new/fuzzy entries) whenever
# you change any documentation source, including English `.md` files under
# $(SOURCEDIR) AND the Python API docstrings picked up by autodoc. Otherwise the
# Chinese docs will fall out of sync with the source.
update-po: gettext
	@sphinx-intl update -p "$(GETTEXTDIR)" -d "$(LOCALEDIR)" -l zh_CN

# Multi-language docs layout: English lives at the site root ($(BUILDDIR)/html/),
# Chinese lives under $(BUILDDIR)/html/zh_CN/. This keeps single- and
# multi-language builds consistent with the language switcher and the CI deploy
# layout ("English at root, Chinese under zh_CN/").
html-en:
	@DOCS_LANGUAGE=en $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

html-zh:
	@DOCS_LANGUAGE=zh_CN $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html/zh_CN" $(SPHINXOPTS) $(O)

html-multilang: html-en html-zh

html: html-multilang

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
