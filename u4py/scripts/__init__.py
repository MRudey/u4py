"""
**Scripts**

This folder contains scripts that use the u4py module for data preparation, analysis and plotting. Most of them require u4py to be installed on the system to work.


| `data_analysis`: Analysis scripts for inversion or other timeseries analysis.
| `data_preparation`: Data preparation for u4py, e.g. converting original data.
| `examples`: Case studies and similar examples.
| `gis_workflows`: Workflows making use of GIS functions
| `helpers`: Data preparation or plotting that is not directly related to u4py.
| `playground`: Some scripts that are either WIP or to explore the data.
| `plotting`: Scripts for plotting or visualizing data, sometimes including analysis.

"""
from . import (
    data_analysis,
    data_preparation,
    examples,
    gis_workflows,
    helpers,
    playground,
    plotting,
)
