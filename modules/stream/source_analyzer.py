import logging
import numpy as np
from modules.detector import Detector
from modules.stream.source_loader import OpencvReader


class SourceAnalyzer:
    def __init__(self, reader: OpencvReader, debug: bool = True):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.debug = debug

        self.reader = reader

    def run(self) -> None:
        self._logger.debug('run SourceAnalyzer')
        while True:
            frames, ids = self.reader.get_frames()
            if len(np.asarray(frames).shape) != 1:
                bbox, class_id = Detector(self.debug).detect_person_bbox(frames)
            for frame, _id in zip(frames, ids):
                if isinstance(frame, np.ndarray):
                    self.reader.show_frames(frame, _id)

