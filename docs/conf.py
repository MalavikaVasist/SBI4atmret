# Configuration file for the Sphinx documentation builder.
# SBI4atmret — Simulation-Based Inference for Atmospheric Retrieval

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
project = "SBI4atmret"
copyright = "2025, Malavika Vasist"
author = "Malavika Vasist"
release = "0.1.0"
version = "0.1"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

# Autosummary
autosummary_generate = True
autosummary_imported_members = False

# Napoleon (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True

# Autodoc
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}
autodoc_typehints = "description"
autodoc_class_signature = "separated"

# Type hints
always_document_param_types = True
typehints_defaults = "comma"

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# Source suffixes
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_title = "SBI4atmret"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "source_repository": "https://github.com/mvasist/SBI4atmret",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- Mock imports for build environment --------------------------------------
autodoc_mock_imports = [
    "petitRADTRANS",
    "lampe",
    "zuko",
    "wandb",
    "dawgz",
    "astropy",
    "scipy",
    "sklearn",
    "h5py",
    "torch",
    "pydantic",
    "numpy",
    "pandas",
    "matplotlib",
    "tqdm",
]
