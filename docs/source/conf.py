# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os.path as osp
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyg_spectral'
copyright = '2024, NTU'
author = 'Ningyi Liao'
release = '1.0.0'

sys.path.insert(0, osp.abspath('../pyg_spectral'))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

autosummary_generate = True
templates_path = ['_templates']
exclude_patterns = []

source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "logo_only": False,
    "display_version": True,
}
html_static_path = ['_static']

html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }
add_module_names = True

html_context = {
    "display_github": True,
    "github_user": "gdmnl",
    "github_repo": "Spectral-GNN-Benchmark",
    "github_version": "main",
    "conf_py_path": "/source/",
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'torch': ('https://pytorch.org/docs/master', None),
    'pyg': ('https://pytorch-geometric.readthedocs.io/en/latest/', None),
}
