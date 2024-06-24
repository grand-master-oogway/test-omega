from __future__ import annotations

import logging
from objects import Config
from .stream.source_loader import OpencvReader
from .stream.source_analyzer import SourceAnalyzer


class Worker:
    def __init__(self, source_list: Config, debug: bool):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self._reader = OpencvReader(source_list, debug)
        self._source_run = SourceAnalyzer(reader=self._reader, debug=debug)

    def run(self):
        self._logger.debug('run Worker')
        self._reader.start()
        self._source_run.run()
