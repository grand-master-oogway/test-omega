from __future__ import annotations

import logging
from objects import Config
from .detector import Detector
from tracker import CentroidTracker
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
        self._detector = Detector(self.config_list.model_config, debug)

    def start(self):
        self._logger.debug('run Worker')
        self._reader.start()
        self._source_run = SourceAnalyzer(reader=self._reader, detector=self._detector, trackers=[CentroidTracker() for _id in range(len(self._reader))], debug=self.debug)

    def run(self):
        self._source_run.run()
