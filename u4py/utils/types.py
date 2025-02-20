"""
Contains types for better type hints in Python
"""

import os
from typing import TypedDict


class U4PathsConfig(TypedDict):
    base_path: os.PathLike
    ext_path: os.PathLike
    places_path: os.PathLike
    output_path: os.PathLike
    psi_path: os.PathLike
    processing_path: os.PathLike
    diff_plan_path: os.PathLike
    u4projects_path: os.PathLike
    tektonik_path: os.PathLike
    bld_path: os.PathLike
    piloten_path: os.PathLike
    base_map_path: os.PathLike
    subsubregions_path: os.PathLike
    psivert_path: os.PathLike
    psiew_path: os.PathLike
    results_path: os.PathLike

class U4Config(TypedDict):
    overwrite: bool
    use_filtered: bool
    use_parallel: bool
    generate_plots: bool
    overwrite_plots: bool
    generate_document: bool
    single_report: bool
    is_hlnug: bool

class U4Metadata(TypedDict):
    report_title: str
    report_subtitle: str
    report_suffix: str

class U4Project(TypedDict):
    paths: U4PathsConfig
    config: U4Config
    metadata: U4Metadata