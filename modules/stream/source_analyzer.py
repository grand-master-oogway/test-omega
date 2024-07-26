from __future__ import annotations

import cv2
import logging
import numpy as np
from typing import List
from modules.detector import Detector
from modules.tracker import CentroidTracker
from modules.stream.source_loader import OpencvReader


class SourceAnalyzer:
    def __init__(self, reader: OpencvReader, detector: Detector, trackers: List[CentroidTracker], debug: bool = True):
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self.debug = debug
        self.reader = reader
        self.detector = detector
        self._trackers = trackers


    def run(self) -> None:
        self._logger.debug('run SourceAnalyzer')
        while True:
            frames, ids = self.reader.get_frames()
            sources_detected_objects = self.detector.get_bbox(frames, ids)

            if sources_detected_objects:
                for frame, _id, detected_objects in zip(frames, ids, sources_detected_objects):
                    detected_objects = self._trackers[_id].update(detected_objects)
                    if isinstance(frame, np.ndarray):
                        for detected_object in detected_objects:
                            unique_id = detected_object.unique_id
                            count = str(detected_object.count)
                            class_name = detected_object.class_name
                            conf = str(detected_object.conf)

                            cv2.rectangle(frame, (detected_object.bbox[0], detected_object.bbox[1]), (detected_object.bbox[2], detected_object.bbox[3]), (0, 255, 0), 2)
                            # cv2.putText(frame, unique_id, (detected_object.bbox[0], detected_object.bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame, count, (detected_object.bbox[0], detected_object.bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame, class_name, (detected_object.bbox[0], detected_object.bbox[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame, conf, (detected_object.bbox[0] + 100, detected_object.bbox[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        self.reader.show_frames(frame, _id)
            else:
                for frame, _id in zip(frames, ids):

                    if isinstance(frame, np.ndarray):
                        self.reader.show_frames(frame, _id)
