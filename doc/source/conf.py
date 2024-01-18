# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "U4Py"
copyright = "2023, M. Rudolf"
author = "M. Rudolf"
release = "0.0.2dev"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
]

templates_path = ["_templates"]
exclude_patterns = ["setup.py"]
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "private-members": True,
    "member-order": "bysource",
    "special-members": "__init__",
}
autodoc_mock_imports = [
    "bmi_arcgis_restapi",
    "contextily",
    "dbfread",
    "ete3",
    "ffmpeg_python",
    "fiona",
    "GDAL",
    "geopandas",
    "geopy",
    "h5py",
    "ipython",
    "matplotlib",
    "mahotas",
    "numba",
    "numba_progress",
    "numpy",
    "openpyxl",
    "osmnx",
    "pandas",
    "pathvalidate",
    "pycwt",
    "pyproj",
    "rasterio",
    "scikit-image",
    "scipy",
    "setuptools",
    "Shapely",
    "tensorflow",
    "tensorflow_intel",
    "tqdm",
    "uncertainties",
    "utm",
]

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
