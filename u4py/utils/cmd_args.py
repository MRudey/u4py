""" Implements the command line arguments """

import argparse
import logging

import u4py.utils.config as u4config


def load(module_descript: str = "No description given."):
    """
    Loads the commandline parser and sets everything in the config for later
    use.
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


def setup_parser(module_descript: str):
    """Sets all command line arguments"""
    parser = argparse.ArgumentParser(description=module_descript)
    parser.add_argument(
        "-i",
        "--input",
        help="Filepath of the input file or folder. You will be asked to provide one when empty.",
        metavar="PATH",
        default="",
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
    """Converts loglevel number into string"""
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
