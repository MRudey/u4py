"""Utility functions for convenience.
"""

import datetime
import logging
import time


def eta(start_time: float, current_progress: float):
    """Logs the proposed remaining time for the process that was achieved since the start time.

    :param start_time: The starting time of the process monitored.
    :type start_time: float
    :param current_progress: The current progress ranging from 0 to 1.
    :type current_progress: float
    """
    if current_progress > 0:
        diff_time = time.time() - start_time
        remaining = (diff_time / current_progress) - diff_time
        eta_time = datetime.datetime.now() + datetime.timedelta(
            seconds=remaining
        )
        logstr = f"{int(remaining)}s remaining, ETA: {eta_time.isoformat()}"
        logging.info(logstr)
        return logstr
