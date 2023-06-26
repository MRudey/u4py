.. U4Py documentation master file, created by
   sphinx-quickstart on Thu Jun 22 10:08:15 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

U4Py Documentation
==================

This Python module was created in the framework of the project Umwelt 4.0 financed by the Hessian Agency for Nature Conservation, Environment and Geology. It is split into several packages:

 - :doc:`analysis <../u4py.analysis>`: Does the data analysis, e.g., inversion or spatial analysis.
 - :doc:`plotting <../u4py.plotting>`: Pre-made figures and axis objects for specific plots including formatting.
 - :doc:`utils <../u4py.utils>`: Various utility functions, e.g., file and project management.
 - :doc:`addons <../u4py.addons>`: Tools to read externally supplied data such as groundwater levels or weather data.

 Each module can be imported and used separately for creating individual workflows and plots.

Installation
============

To install the module you need to use pip together with git:

::

   pip install pip@git+https://git-ce.rwth-aachen.de/rudolf/u4py

This automatically installs the module :doc:`u4py <../u4py>` including all prerequisites to your Python environment.

Examples and Notebooks
======================

Several standalone scripts and interactive Jupyter notebooks are available. Scripts and notebooks require ``u4py`` installed on your system. Additionally, all notebooks and some scripts (``arcpy_*.py``) require a valid installation of ArcGIS including an ``arcpy`` environment.

The scripts contain full workflows to do data preparation or processing. You can find details on the scripts in the :doc:`Documentation of Scripts <../scripts>` and you can `Download Example Scripts <https://git-ce.rwth-aachen.de/rudolf/u4py/-/archive/main/u4py-main.zip?path=scripts>`_ as zip files.

The notebooks contain the workflow to prepare the dataset for manual classification. You can find details in the :doc:`Documentation for Notebooks <../notebooks>` and you can `Download Example Notebooks <https://git-ce.rwth-aachen.de/rudolf/u4py/-/archive/main/u4py-main.zip?path=notebooks>`_ as zip files.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

.. automodule:: u4py
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: scripts
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: notebooks
   :members:
   :undoc-members:
   :show-inheritance:

Indices and tables
==================

* :ref:`Overview of module <modindex>`
* :ref:`Index of all functions <genindex>`
.. * :ref:`search`
