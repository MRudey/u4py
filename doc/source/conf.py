# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "U4Py"
copyright = "2023, T. Treffeisen, M. Rudolf"
author = "T. Treffeisen, M. Rudolf"
release = "0.0.2dev"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc"]
# for x in os.walk(r"C:\Users\Michael Rudolf\HESSENBOX-DA\GitRepos\u4py\u4py"):
#     path = x[0]
#     if ".git" not in path and "__" not in path and "egg-info" not in path:
#         sys.path.insert(0, path)

templates_path = ["_templates"]
exclude_patterns = []

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "classic"
html_static_path = ["_static"]
