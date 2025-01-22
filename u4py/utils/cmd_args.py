"""
Implements the command line arguments. Mainly used for scripts that run on the server or cluster with long-running tasks. Examples are the conversion of a .gpgk file to many hdf5 files or the batch inversion of many hdf5 files on the cluster.
"""

from __future__ import annotations

import argparse
import logging

import u4py.utils.config as u4config


def load(module_descript: str = "No description given."):
    """Loads the commandline parser and sets everything in the config for later use.

    :param module_descript: The description of the module, defaults to "No description given."
    :type module_descript: str, optional
    """
    # Get commandline arguments
    parser = setup_parser(module_descript)
    args = parser.parse_args()

    # Adjust number of cpus
    u4config.cpu_count = args.cpus
    logging.info(f"Using {u4config.cpu_count} CPUs.")

    # Set input path for scripts
    u4config.in_path = args.input
    if u4config.in_path:
        logging.info(f"Input file/folder {u4config.in_path}.")
    else:
        logging.info("No input given. Asking user.")

    # Adapt logger level when set to verbose (-v)
    if args.verbose:
        u4config.log_level = logging.DEBUG
        logging.getLogger().setLevel(u4config.log_level)
    logging.info(
        f"Set loglevel to {log_level_str(logging.getLogger().level)}."
    )
    return args


def setup_parser(module_descript: str) -> argparse.ArgumentParser:
    """Sets all command line arguments

    :param module_descript: The description of the module.
    :type module_descript: str
    :return: The argument parser for the module.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description=module_descript)
    parser.add_argument(
        "-i",
        "--input",
        help="Filepath of the config file. You will be asked to provide one when empty.",
        metavar="PATH",
        default="",
    )
    parser.add_argument(
        "-ov",
        "--overwrite",
        help="Overwrite results and intermediate files.",
        default=False,
        type=bool,
    )
    parser.add_argument(
        "-c",
        "--cpus",
        help="Number of CPUs to use for parallel processing.",
        metavar="N",
        default=u4config.cpu_count,
        type=int,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        help="Sets the logging level to DEBUG.",
        action="store_true",
    )
    return parser


def log_level_str(num: int) -> str:
    """Converts loglevel number into string

    :param num: The number of the loglevel.
    :type num: int
    :return: The string of the loglevel.
    :rtype: str
    """
    if num == 50:
        return "CRITICAL"
    elif num == 40:
        return "ERROR"
    elif num == 30:
        return "WARNING"
    elif num == 20:
        return "INFO"
    elif num == 10:
        return "DEBUG"
    elif num == 0:
        return "NOTSET"
