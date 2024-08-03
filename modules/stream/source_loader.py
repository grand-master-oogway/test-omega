from __future__ import annotations

import cv2
import time
import logging
import threading
from objects import Config, Source
from typing import List, Tuple, Union


class OpencvReader:
    """
    Class for source read.

    """
    _TIME = 1

    def __init__(self, sources_data: Config, debug: bool = True):
        """
        Constructor for source read.

        Args:
            sources_data (Config): Configuration data containing source information.
            debug (bool): Flag to set the logging level to DEBUG if True, otherwise INFO.
        """
        self._logger: logging.Logger = logging.getLogger(type(self).__name__)
        self._logger.setLevel(logging.DEBUG if debug else logging.INFO)

        self._sources: List[Source] = sources_data.sources

    def __len__(self) -> int:
        """
        Get the number of sources.

        Returns:
            int: The number of sources.
        """
        return len(self._sources)

    def start(self) -> OpencvReader:
        """
        Start the thread for each source.

        Returns:
            OpencvReader: The current instance.
        """
        self._logger.debug('create thread for each source')
        for source in self._sources:
            threading.Thread(target=self._update, args=(source,), daemon=True).start()

        return self

    def get_frames(self) -> Tuple[List[np.ndarray], List[int]]:
        """
        Get the frames from all sources.

        Returns:
            Tuple[List[np.ndarray], List[int]]: A tuple containing a list of frames and a list of source IDs.
        """
        frames, ids = [], []
        for i, source in enumerate(self._sources):
            frames.append(source.frame)
            ids.append(i)
        return frames, ids

    def _initialize_capture(self, source: Union[str, int]) -> cv2.VideoCapture:
        """
        Initialize the video capture for a given source.

        Args:
            source (Union[str, int]): The source URL or device index.

        Returns:
            cv2.VideoCapture: The video capture object.
        """
        capture = cv2.VideoCapture(source)
        while True:
            if not capture.isOpened():
                capture.release()
                time.sleep(self._TIME)
                capture = cv2.VideoCapture(source)
            return capture

    def _update(self, source: Source) -> None:
        """
        Update the frame for a given source.

        Args:
            source (Source): The source object containing the RTSP URL and frame.
        """
        while True:
            capture = self._initialize_capture(source.rtsp)
            if not capture.isOpened():
                capture = self._initialize_capture(source.rtsp)
            _read, frame = capture.read()
            if not _read:
                capture = self._initialize_capture(source.rtsp)

            source.frame = frame

    @staticmethod
    def show_frames(frame: np.ndarray, number: int) -> None:
        """
        Show the frame in a window.

        Args:
            frame (np.ndarray): The frame to show.
            number (int): The window number.
        """
        cv2.imshow(f'{number}', frame)
        cv2.waitKey(1)
