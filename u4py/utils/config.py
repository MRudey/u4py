"""
Default configuration for some scripts and modules.
"""
from __future__ import annotations

import logging
import os

cpu_count = os.cpu_count() - 4
in_path = ""
log_level = logging.INFO
