# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os.path as osp
import sys
from datetime import datetime
import importlib
import inspect
# import pyg_sphinx_theme

import pyg_spectral

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

proot = osp.abspath(osp.dirname(osp.dirname(osp.dirname((osp.realpath(__file__))))))
sys.path.insert(0, proot)
sys.path.insert(1, osp.join(proot, 'benchmark'))
# sys.path.append(osp.join(osp.dirname(pyg_sphinx_theme.__file__), 'extension'))

project = 'pyg_spectral'
author = 'Ningyi Liao'
copyright = f'{datetime.now().year}, NTU'
version = pyg_spectral.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

templates_path = ['_templates']
exclude_patterns = []
source_suffix = ['.rst', '.md']

add_module_names = False
show_authors = False
modindex_common_prefix = ['pyg_spectral.', 'benchmark.']

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.linkcode',
    'nbsphinx',
    'myst_parser',
    'sphinx_copybutton',
    'sphinx_autodoc_typehints',
    # 'pyg',
]

autodoc_member_order = 'bysource'
autodoc_typehints = 'signature'
autosummary_generate = True
autosummary_imported_members = True
copybutton_exclude = '.linenos, .gp'

napoleon_numpy_docstring = False
napoleon_custom_sections = [
    ('Returns', 'params_style'),
    ('Updates', 'params_style'),
]
napoleon_use_rtype = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_preprocess_types = True
autodoc_type_aliases = {
    "Tensor": ":external:class:`Tensor <torch.Tensor>`",
    "SparseTensor": ":external:func:`SparseTensor <torch.sparse_csr_tensor>`",
    "pyg": "torch_geometric",
}
napoleon_type_aliases = autodoc_type_aliases
always_use_bars_union = True
typehints_defaults = 'comma'
typehints_document_rtype = False
typehints_use_signature = True
typehints_use_signature_return = True

latex_elements = {
    'preamble': r'''
\usepackage[utf8]{inputenc}
\usepackage{charter}
\usepackage[defaultsans]{lato}
''',
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "navigation_depth": 2,
    "prev_next_buttons_location": "both",
}
html_static_path = ['_static']

html_sidebars = { '**': ['globaltoc.html', 'relations.html', 'sourcelink.html', 'searchbox.html'] }

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
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('http://pandas.pydata.org/pandas-docs/stable/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'pyg': ('https://pytorch-geometric.readthedocs.io/en/latest/', None),
    'torch_geometric': ('https://pytorch-geometric.readthedocs.io/en/latest/', None),
    'torchmetrics': ('https://lightning.ai/docs/torchmetrics/stable/', None),
}

code_url = f"https://github.com/gdmnl/Spectral-GNN-Benchmark/blob/main"

def linkcode_resolve(domain, info):
    def _resolve_python(info):
        # ref: https://github.com/scikit-learn/scikit-learn/blob/f7026e575a494b47b50557932b5c3ce0688fdc72/doc/sphinxext/github_link.py
        def _inspect_macro(obj, func):
            try:
                res = func(obj)
            except Exception:
                res = None
            if not res:
                try:
                    res = func(sys.modules[obj.__module__])
                except Exception:
                    res = None
            if not res:
                for f in ['fget', 'fset', 'fdel']:
                    try:
                        res = func(getattr(obj, f))
                        break
                    except Exception:
                        continue
            return res

        mod = importlib.import_module(info["module"])
        is_parent = False
        if "." in info["fullname"]:
            objname, attrname = info["fullname"].split(".")
            obj = getattr(mod, objname)
            try:
                obj = getattr(obj, attrname)
            except AttributeError:
                # object is a class attribute
                is_parent = True
        else:
            obj = getattr(mod, info["fullname"])
        obj = inspect.unwrap(obj)

        file = _inspect_macro(obj, inspect.getsourcefile)
        if not file:
            return
        file = osp.relpath(file, proot)

        lines = _inspect_macro(obj, inspect.getsourcelines)
        if is_parent or not lines:
            return f"{code_url}/{file}"
        start, end = lines[1], lines[1] + len(lines[0]) - 1
        return f"{code_url}/{file}#L{start}-L{end}"

    if domain in ["py"]:
        if not info.get("module") or not info.get("fullname"):
            return None
        return _resolve_python(info)
    elif domain in ["c", "cpp"]:
        if not info.get("names"):
            return None
        # TODO: https://www.sphinx-doc.org/en/master/usage/extensions/linkcode.html
        pass
    return None
