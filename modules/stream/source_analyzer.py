import logging
import numpy as np
from modules.detector import Detector
from modules.stream.source_loader import OpencvReader


class SourceAnalyzer:
    def __init__(self, reader: OpencvReader, debug: bool = True):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self.reader = reader

    # def run(self) -> None:
    #     self._logger.debug('run SourceAnalyzer')
    #     res = None
    #     while True:
    #         frames, ids = self.reader.get_frames()
    #         Detector(frames, ids).run()
    #         for frame, _id in zip(frames, ids):
    #             if isinstance(frame, np.ndarray):
    #                 print("frame --> ", frame)
    #                 self.reader.show_frames(frame, _id)

    def run(self) -> None:
        self._logger.debug('run SourceAnalyzer')
        res = None
        while True:
            frames, ids = self.reader.get_frames()
            if len(np.asarray(frames).shape) == 1:
                continue
            # print(frames)
            # print(np.asarray(frames).shape)
            self.reader.show_frames(Detector(frames, ids).run(), 0)

