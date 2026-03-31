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

.PHONY: gettext update-po html-en html-zh html-multilang

gettext:
	@$(SPHINXBUILD) -M gettext "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

update-po: gettext
	@sphinx-intl update -p "$(GETTEXTDIR)" -d "$(LOCALEDIR)" -l zh_CN

html-en:
	@DOCS_LANGUAGE=en $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html/en" $(SPHINXOPTS) $(O)

html-zh:
	@DOCS_LANGUAGE=zh_CN $(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html/zh_CN" $(SPHINXOPTS) $(O)

html-multilang: html-en html-zh

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
