This directory stores Sphinx i18n translation catalogs.

Recommended workflow:

1. Run `make update-po` in `docs/`.
2. Edit `zh_CN/LC_MESSAGES/*.po` files and fill `msgstr`.
3. Run `make html-multilang` to build `en` and `zh_CN` docs.
