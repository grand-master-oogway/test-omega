import logging
import numpy as np
from modules.detector import Detector
from modules.stream.source_loader import OpencvReader


class SourceAnalyzer:
    def __init__(self, reader: OpencvReader, debug: bool = True):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.reader = reader

    def run(self) -> None:
        self._logger.debug('run SourceAnalyzer')
        res = None
        while True:
            frames, ids = self.reader.get_frames()
            if len(np.asarray(frames).shape) != 1:
                bbox = Detector(frames, ids).detect_person_bbox()
            for frame, _id in zip(frames, ids):
                if isinstance(frame, np.ndarray):
                    self.reader.show_frames(frame, _id)

