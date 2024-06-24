import logging
import numpy as np
from modules.stream.source_loader import OpencvReader


class SourceAnalyzer:
    def __init__(self, reader: OpencvReader, debug: bool = True):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.reader = reader

    def run(self):
        self._logger.debug('run SourceAnalyzer')
        while True:
            frames = self.reader.get_frames()
            for frame in frames:
                if isinstance(frame, np.ndarray):
                    self.reader.show_frames(frame)
