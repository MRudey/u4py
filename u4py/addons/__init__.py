"""
**Package for additional data import and preparation.**

Usually other data is supplied in various file formats and also sometimes
formatted in non-standard ways. The submodules in this package implement
specific procedures to import and convert data in formats that are not easily
read by `GeoPandas` or `NumPy`. In some cases the data has to be converted to
match the other datatypes in our projects.
"""
from . import climate, gas_storage, geology, groundwater, rivers, web_services
from .climate import *
from .gas_storage import *
from .geology import *
from .groundwater import *
from .rivers import *
from .web_services import *
