from __future__ import annotations

from objects import Config
from .stream.source_loader import OpencvReader
from .stream.source_analyzer import SourceAnalyzer


class Worker:
    def __init__(self, source_list: Config, debug: bool):
        self._reader = OpencvReader(source_list, debug)
        self._source_run = SourceAnalyzer(reader=self._reader)

    def run(self):
        self._reader.start()
        self._source_run.run()
