"""
Default configuration for some scripts and modules.
"""
from __future__ import annotations

import logging
import os
import sys

cpu_count = os.cpu_count() - 4
in_path = ""
log_level = logging.INFO


def start_logger():
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(funcName)s: %(message)s",
        stream=sys.stdout,
        level=log_level,
    )
