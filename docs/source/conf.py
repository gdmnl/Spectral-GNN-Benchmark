# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os.path as osp
import sys
# import pyg_sphinx_theme

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pyg_spectral'
copyright = '2024, NTU'
author = 'Ningyi Liao'
release = '1.0.0'

proot = osp.abspath(osp.dirname(osp.dirname(osp.dirname((osp.realpath(__file__))))))
sys.path.insert(0, proot)
sys.path.insert(1, osp.join(proot, 'benchmark'))
# sys.path.append(osp.join(osp.dirname(pyg_sphinx_theme.__file__), 'extension'))

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
    'myst_parser',
    # 'pyg',
]

autosummary_generate = True
autosummary_imported_members = True
templates_path = ['_templates']
exclude_patterns = []

source_suffix = ['.rst', '.md']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "navigation_depth": 2,
}
html_static_path = ['_static']

html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }
add_module_names = True

html_context = {
    "display_github": True,
    "github_user": "gdmnl",
    "github_repo": "Spectral-GNN-Benchmark",
    "github_version": "main",
    "conf_py_path": "/docs/source/",
}

autodoc_default_options = {
    'members': True,
    'private-members': True,
    'undoc-members': True,
    'show_headings': False,
}

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'torch': ('https://pytorch.org/docs/master', None),
    'pyg': ('https://pytorch-geometric.readthedocs.io/en/latest/', None),
}
