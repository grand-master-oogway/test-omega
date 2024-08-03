from __future__ import annotations

import numpy as np
from uuid import uuid4
from typing import List, OrderedDict, Dict
from collections import OrderedDict
from scipy.spatial import distance as dist

from objects import DetectedObject


class CentroidTracker:
    """
    Class for tracking objects.
    """
    def __init__(self, maxDisappeared: int = 30, maxDistance: int = 200):

        """
        Constructor for CentroidTracker class.

        Args:
            maxDisappeared (int): Maximum number of consecutive frames an object can be missing before deregistration.
            maxDistance (int): Maximum distance between centroids to associate an object.
        """
        self._maxDistance: int = maxDistance
        self._maxDisappeared: int = maxDisappeared

        self._nextObjectID: str = str(uuid4())
        self._objects: OrderedDict[str, DetectedObject] = OrderedDict()
        self._disappeared: Dict[str, int] = OrderedDict()

    def _register(self, obj: DetectedObject) -> None:
        """
        Register a new object.

        Args:
            obj (DetectedObject): The object to be registered.
        """
        obj.count = 0
        obj.unique_id = self._nextObjectID
        self._objects[self._nextObjectID] = obj
        self._disappeared[self._nextObjectID] = 0
        self._nextObjectID = str(uuid4())

    def _deregister(self, objectID: str) -> None:
        """
        Deregister an object.

        Args:
            objectID (str): The ID of the object to be deregistered.
        """
        del self._objects[objectID]
        del self._disappeared[objectID]

    def update(self, detection_objects: List[DetectedObject]) -> List[DetectedObject]:
        """
        Update the tracker with new detected objects.

        Args:
            detection_objects (List[DetectedObject]): List of detected objects.

        Returns:
            List[DetectedObject]: List of updated detected objects.
        """
        if len(detection_objects) == 0:
            for objectID in list(self._disappeared.keys()):
                self._disappeared[objectID] += 1

                if self._disappeared[objectID] > self._maxDisappeared:
                    self._deregister(objectID)

            return detection_objects

        if len(self._objects) == 0:
            for detection_object in detection_objects:
                self._register(detection_object)
        else:
            objectIDs: List[str] = list(self._objects.keys())
            D: np.ndarray = dist.cdist(
                np.array([detection_object.centroid for detection_object in self._objects.values()]),
                np.array([detection_object.centroid for detection_object in detection_objects])
            )

            rows: np.ndarray = D.min(axis=1).argsort()
            cols: np.ndarray = D.argmin(axis=1)[rows]

            usedRows: set = set()
            usedCols: set = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue

                objectID: str = objectIDs[row]
                if D[row, col] > self._maxDistance:
                    continue

                detection_objects[col].count = self._objects[objectID].count + 1
                detection_objects[col].unique_id = objectID
                self._objects[objectID] = detection_objects[col]
                self._disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows: set = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols: set = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self._disappeared[objectID] += 1

                if self._disappeared[objectID] > self._maxDisappeared:
                    self._deregister(objectID)

            for col in unusedCols:
                self._register(detection_objects[col])

        return detection_objects
