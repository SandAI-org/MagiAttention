# Copyright (c) 2025 SandAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../../"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "MagiAttention"
copyright = "2025, Sandai"
author = "Sandai"
release = "main"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "myst_parser",
    "sphinx_copybutton",
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

autodoc_default_options = {
    "members": None,
    "inherited-members": None,
    "show-inheritance": True,
    "imported-members": False,
    "special-members": False,
    "exclude-members": "__weakref__,__dict__,__module__",
    "private-members": False,
    "member-order": "bysource",
    "undoc-members": False,
}

autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
pygments_style = "colorful"

templates_path = ["_templates"]
exclude_patterns = []  # type: ignore

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_baseurl = "https://sandai-org.github.io/MagiAttention/docs/"
html_show_sourcelink = False

html_theme_options = {
    "show_nav_level": 1,
    "show_toc_level": 3,
    "navigation_depth": 4,
    "collapse_navigation": False,
    "show_version_warning_banner": True,
    "logo": {
        "image_light": "_static/sand-logos/magi-black.png",
        "image_dark": "_static/sand-logos/magi-black.png",
        "text": "MagiAttention",
    },
    "show_prev_next": False,
    "navbar_start": ["navbar-logo", "version-switcher"],
    "switcher": {
        "json_url": (
            "https://raw.githubusercontent.com/SandAI-org/MagiAttention/"
            "refs/heads/gh-pages/docs/versions.json"
        ),
        "version_match": "main",
    },
    "icon_links": [
        {
            "name": "Github",
            "url": "https://github.com/SandAI-org/MagiAttention",
            "icon": "fa-brands fa-github",
        },
        {
            "name": "Blog",
            "url": "https://sandai-org.github.io/MagiAttention/blog/",
            "icon": "fas fa-book-bookmark",
            "type": "fontawesome",
        },
    ],
}


def skip_signature(app, what, name, obj, options, signature, return_annotation):
    return "", None


def setup(app):
    app.connect("autodoc-process-signature", skip_signature)
    app.add_css_file("custom.css")
