from __future__ import annotations

import logging
from typing import Union, List

from objects import Config
from .handler import Handler
from .detector import Detector
from tracker import CentroidTracker
from .stream.source_loader import OpencvReader
from .stream.source_analyzer import SourceAnalyzer


class Worker:
    """
    Class for start and run script.

    """
    def __init__(self, config_list: Config, debug: bool):
        """
        Constructor for Worker class.

        Args:
            config_list (Config): Configuration object containing necessary parameters.
            debug (bool): Flag to set the logging level to DEBUG if True, otherwise INFO.
        """
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.debug: bool = debug
        self._source_run: Union[SourceAnalyzer, None] = None
        self.config_list: Config = config_list
        self._reader: OpencvReader = OpencvReader(self.config_list, debug)
        self._detector: Detector = Detector(self.config_list.model_config, debug)

    def start(self) -> None:
        """
        Start the Worker, initializing the source reader and analyzer.
        """
        self._logger.debug('run Worker')
        self._reader.start()
        self._source_run = SourceAnalyzer(
            reader=self._reader,
            detector=self._detector,
            trackers=[CentroidTracker() for _ in range(len(self._reader))],
            handlers=[Handler() for _ in range(len(self._reader))],
            debug=self.debug
        )

    def run(self) -> None:
        """
        Run the source analyzer.
        """
        if self._source_run:
            self._source_run.run()
