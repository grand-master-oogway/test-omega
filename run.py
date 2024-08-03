from __future__ import annotations

import json
import logging
import argparse

from objects import Config
from modules.worker import Worker
from argparse import ArgumentParser
from logger_configuration import LoggerConfigurator


def parsing() -> ArgumentParser:
    """
    Sets up the argument parser to handle command-line inputs.

    Returns:
        ArgumentParser: Configured argument parser with defined arguments.
    """
    parser = argparse.ArgumentParser(description='Process some sources')
    parser.add_argument('--config', '-c', type=str, nargs='?', help='configure file processing', required=True)
    parser.add_argument('--debug', '-d', type=bool, nargs='?', help='debug true/false', required=True)
    return parser


def _read_config(path: str) -> Config:
    """
    Reads the configuration file from the specified path and initializes a Config object.

    Args:
        path (str): Path to the configuration file.

    Returns:
        Config: Initialized Config object with attributes set from the configuration file.
    """
    with open(path) as f:
        config = Config(json.load(f))
        config.get_attributes
    return config


def main() -> None:
    """
    Main entry point of the script. Parses command-line arguments, configures logging,
    initializes the worker with the configuration, and starts the worker for processing.
    """
    args = parsing().parse_args()

    LoggerConfigurator().set_level_warning().add_console().add_timed_rotating()
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    logger.info('test logger')

    worker: Worker = Worker(_read_config(args.config), args.debug)
    worker.start()
    worker.run()


if __name__ == '__main__':
    """
    Ensures the main function runs when the script is executed directly.
    """
    main()
