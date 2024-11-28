# u4py

This repository contains an installable Python module, Jupyter notebooks and scripts for the Umwelt 4.0 project together with HLNUG. The project is about the semi-automatic detection and classification of ground motions in Hessen.

## Installation

### Prerequisites

Currently supported Python **Version: 3.11.**, due to some packages being not available for 3.12, yet.

Some scripts use ADAtools by Barra et al. 2017, Tomás et al. 2019 and Navarro et al. 2020. If this functionality is needed, please contact one of the authors for a personal copy of ADAtools.

### Setup

To install the module you need to use pip together with git:

```bash
pip install git+https://git-ce.rwth-aachen.de/rudolf/u4py
```

This automatically installs the module u4py including all prerequisites to your Python environment.

## Documentation

You can find an automatically generated documentation [here](https://rudolf.pages.git-ce.rwth-aachen.de/u4py/).

Currently, the python module `u4py` and the `scripts` are mainly for the analysis of persistent scatterer interferometry data (PSI). The Jupyter `notebooks` additionally use digital elevation models, height differences and further information.

## References

- A. Barra, L. Solari, M. Béjar-Pizarro, O. Monserrat, S. Bianchini, G. Herrera, M. Crosetto, R. Sarro, A. G.E., R. Maria Mateos, S. Ligüerzana, C. López, S. Moretti, A Methodology to Detect and Update Active Deformation Areas Based on Sentinel-1 SAR Images , Remote Sensing, Vol. 9, No. 10, September 2017.

- R. Tomás, J. Ignacio Pagán, J. A. Navarro, M. Cano, J. Luis Pastor, A. Riquelme, M. Cuevas-González, M. Crosetto, A. Barra, O. Monserrat, J. M. López-Sánchez, A. Ramón, S. Iborra, M. del Soldato, L. Solari, S. Bianchini, F. Raspini, F. Novali, A. Ferreti, M. Constantini, F. Trillo, G. Herrera, N. Casagli, Semi-Automatic Identification and Pre-Screening of Geological-Geotechnical Deformational Processes Using Persistent Scatterer Interferometry Datasets , Remote Sensing, Vol. 11, No. 14, July 2019.

- J. A. Navarro, R. Tomás, A, Barra, J. I. Pagán, C. Reyes-Carmona, L. Solari, J. L.Vinielles, S. Falco, M. Crosetto, ADAtools: Automatic Detection and Classification of Active Deformation Areas from PSI Displacement Maps. ISPRS Int. J. Geo-Inf. 2020, 9, 584.
