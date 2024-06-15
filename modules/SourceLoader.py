from __future__ import annotations

import cv2
import time
import threading
from typing import List, Tuple


class OpencvReader():
    _TIME = 1

    def __init__(self, sources: List):
        self._sources = sources

    def start(self) -> OpencvReader:
        for source in self._sources:
            threading.Thread(target=self._update, args=(source,), daemon=True).start()

        return self

    def get_frames(self) -> Tuple[List, list]:
        frames, ids = [], []
        for i, source in enumerate(self._sources):
            frames.append(source.frame)
            ids.append(i)
        return frames, ids

    def _initialize_capture(self, source) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(source)
        while True:
            if not capture.isOpened():
                capture.release()
                time.sleep(self._TIME)
                capture = cv2.VideoCapture(source)

            return capture

    def _update(self, source) -> None:
        while True:
            capture = self._initialize_capture(source.rtsp)
            if not capture.isOpened():
                capture = self._initialize_capture(source.rtsp)
            _read, frame = capture.read()
            if not _read:
                capture = self._initialize_capture(source.rtsp)
            source.frame = frame

    @staticmethod
    def show_frames(frame, number) -> None:
        cv2.imshow(f'{number}', frame)
        cv2.waitKey(1)
