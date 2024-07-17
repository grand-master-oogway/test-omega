from __future__ import annotations

import logging
from objects import Config
from .stream.source_loader import OpencvReader
from .stream.source_analyzer import SourceAnalyzer


class Worker:
    def __init__(self, config_list: Config, debug: bool):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.debug = debug
        self._source_run = None
        self.config_list = config_list
        self._reader = OpencvReader(self.config_list, debug)

    def start(self):
        self._logger.debug('run Worker')
        self._reader.start()
        self._source_run = SourceAnalyzer(reader=self._reader, model_config=self.config_list.model_config, debug=self.debug)

    def run(self):
        self._source_run.run()
