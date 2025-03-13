# Configuration file for the Sphinx documentation builder
#
# For the full list of built-in configuration options see:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
project = 'BRIDGE'
copyright = '2025, Jonathan Rubin'
author = 'Jonathan Rubin, Sahil Loomba, Nick S. Jones'
version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',         # Auto-generate API docs
    'sphinx.ext.napoleon',        # Support for Google-style docstrings
    'sphinx.ext.mathjax',         # Math support
    'sphinx.ext.viewcode',        # Link to source code
    'sphinx.ext.intersphinx',     # Link to other documentation
    'sphinx.ext.autosectionlabel', # Allow reference sections
    'sphinx_rtd_theme',           # Read the Docs theme
    'recommonmark',               # Support for Markdown
]

# Add source files extension
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# The master toctree document
master_doc = 'index'

# List of patterns to exclude from source
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'navigation_depth': 4,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'includehidden': True,
    'titles_only': False,
    'display_version': True,
    'logo_only': False,
}

# Add any paths that contain templates
templates_path = ['_templates']

# Add any paths that contain custom static files
html_static_path = ['_static']

# Custom sidebar templates
html_sidebars = {
    '**': [
        'relations.html',  # needs 'show_related': True theme option
        'searchbox.html',
    ]
}

# Output file base name for HTML help builder
htmlhelp_basename = 'BRIDGEdoc'

# -- Extension configuration -------------------------------------------------
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'
autoclass_content = 'both'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable', None),
    'torch': ('https://pytorch.org/docs/stable', None),
    'dgl': ('https://docs.dgl.ai/en/latest', None),
}

# -- Additional setup --------------------------------------------------------
def setup(app):
    app.add_css_file('custom.css')
