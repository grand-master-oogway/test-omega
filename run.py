from __future__ import annotations

import json
import logging
import argparse
from objects import Config
from modules.worker import Worker
from argparse import ArgumentParser
from logger_configuration import LoggerConfigurator


def parsing() -> ArgumentParser:
    parser = argparse.ArgumentParser(description='Process some sources')
    parser.add_argument('--config', '-c', type=str, nargs='?', help='configure file processing', required=True)

    parser.add_argument('--debug', '-d', type=bool, nargs='?', help='debug true/false', required=True)

    return parser


def _read_config(path: str) -> Config:
    with open(path) as f:
        data = json.load(f)
        config = Config(list(data['sources_data'].values())[0])
    return config


def main():
    args = parsing().parse_args()

    LoggerConfigurator().set_level_warning().add_console().add_timed_rotating()
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    logger.info('test logger')
    Worker(_read_config(args.config), args.debug).run()


if __name__ == '__main__':
    main()
