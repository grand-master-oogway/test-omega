from __future__ import annotations

import logging
import numpy as np
from typing import List
from modules.detector import Detector
from modules.tracker import CentroidTracker
from objects.model_config import ModelConfig
from modules.stream.source_loader import OpencvReader


class SourceAnalyzer:
    def __init__(self, reader: OpencvReader, model_config: ModelConfig, debug: bool = True):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.debug = debug
        self.reader = reader
        self.model_config = model_config


    def run(self) -> None:
        self._logger.debug('run SourceAnalyzer')
        while True:
            frames, ids = self.reader.get_frames()
            trackers = [CentroidTracker() for _id in ids]
            detected_objects = Detector(self.model_config, self.debug).get_bbox(frames, ids)
            if detected_objects:
                for frame, _id, detected_object in zip(frames, ids, detected_objects):
                    trackers[_id].update(detected_object)
                    if isinstance(frame, np.ndarray):
                        self.reader.show_frames(frame, _id)
            else:
                for frame, _id in zip(frames, ids):

                    print(frame)
                    if isinstance(frame, np.ndarray):
                        self.reader.show_frames(frame, _id)
