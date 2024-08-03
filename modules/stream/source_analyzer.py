from __future__ import annotations

import cv2
import logging
import numpy as np
from typing import List

from modules.handler import Handler
from modules.detector import Detector
from modules.tracker import CentroidTracker
from objects.detected_object import DetectedObject
from modules.stream.source_loader import OpencvReader


class SourceAnalyzer:
    """
    Class for source analyze objects.

    """
    def __init__(self, reader: OpencvReader, detector: Detector, trackers: List[CentroidTracker], handlers: List[Handler], debug: bool = True):
        """
        Constructor for source analyze class.

        Args:
            reader (OpencvReader): The reader object for capturing video frames.
            detector (Detector): The detector object for detecting objects in frames.
            trackers (List[CentroidTracker]): A list of CentroidTracker objects for tracking detected objects.
            handlers (List[Handler]): A list of Handler objects for handling detected objects.
            debug (bool): Flag to set the logging level to DEBUG if True, otherwise INFO.
        """
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)
        self._reader: OpencvReader = reader
        self._detector: Detector = detector
        self._trackers: List[CentroidTracker] = trackers
        self._handlers: List[Handler] = handlers

    def _detect2frame(self, frame: np.ndarray, detected_objects: List[DetectedObject]) -> None:
        """
        Draw bounding boxes and labels on the frame for detected objects.

        Args:
            frame (np.ndarray): The frame to draw on.
            detected_objects (List[DetectedObject]): A list of detected objects.
        """
        for detected_object in detected_objects:
            count = str(detected_object.count)
            class_name = detected_object.class_name
            conf = str(detected_object.conf)
            output = f'{class_name}  {conf}'

            cv2.rectangle(frame, (detected_object.bbox[0], detected_object.bbox[1]),
                          (detected_object.bbox[2], detected_object.bbox[3]), (0, 255, 0), 2)
            cv2.putText(frame, count, (detected_object.bbox[0], detected_object.bbox[1] + 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)
            cv2.putText(frame, output, (detected_object.bbox[0], detected_object.bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    def run(self) -> None:
        """
        Run the source analyzer to process video frames and detect objects.
        """
        self._logger.debug('run SourceAnalyzer')
        while True:
            frames, ids = self._reader.get_frames()
            sources_detected_objects = self._detector.get_bbox(frames, ids)

            if sources_detected_objects:
                if not len(sources_detected_objects):
                    continue
                for frame, _id, detected_objects in zip(frames, ids, sources_detected_objects):
                    detected_objects = self._trackers[_id].update(detected_objects)
                    if isinstance(frame, np.ndarray):
                        self._detect2frame(frame, detected_objects)
                        self._reader.show_frames(frame, _id)
                    detected_objects = self._handlers[_id].update(detected_objects)

                    for detected_object in detected_objects:
                        print(f'object with id: {detected_object.unique_id} and class_id: {detected_object.class_name} is sent')
