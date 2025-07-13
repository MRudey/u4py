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
 - :doc:`io <../u4py.io>`: Tools to read various file formats.
 - :doc:`addons <../u4py.addons>`: Tools to read externally supplied data such as groundwater levels or weather data.

:doc:`View the Full API Documentation here <u4py>`

Each module can be imported and used separately for creating individual workflows and plots.

Installation
============
Currently supported Python Versions: 3.10. to 3.13.

Some scripts use ADAtools by Barra et al. 2017, Tomás et al. 2019 and Navarro et al. 2020. If this functionality is needed, please contact one of the authors for a personal copy of ADAtools.

To install the module you need to use pip together with git:

::

   python -m pip install git+https://git-ce.rwth-aachen.de/rudolf/u4py

This automatically installs the module :doc:`u4py <../u4py>` including all prerequisites to your Python environment.

Examples and Notebooks
======================

Several standalone scripts and interactive Jupyter notebooks are available. Scripts and notebooks require ``u4py`` installed on your system. Additionally, all notebooks and some scripts (``arcpy_*.py``) require a valid installation of ArcGIS including an ``arcpy`` environment.

The scripts contain full workflows to do data preparation or processing. You can `Download Example Scripts <https://git-ce.rwth-aachen.de/rudolf/u4py/-/archive/main/u4py-main.zip?path=scripts>`_ as zip files.

Source
======

You can find the `source repository here: <https://git-ce.rwth-aachen.de/rudolf/u4py>`_.


Indices and tables
==================

* :ref:`Overview of module <modindex>`
* :ref:`Index of all functions <genindex>`

References
----------

- A. Barra, L. Solari, M. Béjar-Pizarro, O. Monserrat, S. Bianchini, G. Herrera, M. Crosetto, R. Sarro, A. G.E., R. Maria Mateos, S. Ligüerzana, C. López, S. Moretti, A Methodology to Detect and Update Active Deformation Areas Based on Sentinel-1 SAR Images , Remote Sensing, Vol. 9, No. 10, September 2017.

- R. Tomás, J. Ignacio Pagán, J. A. Navarro, M. Cano, J. Luis Pastor, A. Riquelme, M. Cuevas-González, M. Crosetto, A. Barra, O. Monserrat, J. M. López-Sánchez, A. Ramón, S. Iborra, M. del Soldato, L. Solari, S. Bianchini, F. Raspini, F. Novali, A. Ferreti, M. Constantini, F. Trillo, G. Herrera, N. Casagli, Semi-Automatic Identification and Pre-Screening of Geological-Geotechnical Deformational Processes Using Persistent Scatterer Interferometry Datasets , Remote Sensing, Vol. 11, No. 14, July 2019.

- J. A. Navarro, R. Tomás, A, Barra, J. I. Pagán, C. Reyes-Carmona, L. Solari, J. L.Vinielles, S. Falco, M. Crosetto, ADAtools: Automatic Detection and Classification of Active Deformation Areas from PSI Displacement Maps. ISPRS Int. J. Geo-Inf. 2020, 9, 584.
