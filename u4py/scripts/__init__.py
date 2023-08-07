"""
**Scripts**

This folder contains scripts that use the u4py module for data preparation, analysis and plotting. Most of them require u4py to be installed on the system to work.


| `helpers`: Data preparation or plotting that is not directly related to u4py.
| `data_preparation`: Data preparation for u4py, e.g. converting original data.
| `data_analysis`: Analysis scripts for inversion or other timeseries analysis.
| `plotting`: Scripts for plotting or visualizing data, sometimes including analysis.
| `playground`: Some scripts that are either WIP or to explore the data.

"""
from . import data_analysis, data_preparation, helpers, playground, plotting
