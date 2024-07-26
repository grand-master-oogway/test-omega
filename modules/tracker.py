from __future__ import annotations

import numpy as np
from uuid import uuid4
from typing import List, Tuple
from collections import OrderedDict
from scipy.spatial import distance as dist

from objects import DetectedObject


class CentroidTracker:
    def __init__(self, maxDisappeared=30, maxDistance=200):
        self.nextObjectID = str(uuid4())
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()

        self.maxDisappeared = maxDisappeared
        self.maxDistance = maxDistance


    def register(self, obj: DetectedObject) -> None:
        obj.count = 0
        obj.unique_id = self.nextObjectID
        self.objects[self.nextObjectID] = [obj]
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID = str(uuid4())

    def deregister(self, objectID) -> None:
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, detection_objects: List[DetectedObject]) -> List[DetectedObject]:

        if len(detection_objects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            return detection_objects

        if len(self.objects) == 0:
            for detection_object in detection_objects:
                self.register(detection_object)


        else:
            objectIDs = list(self.objects.keys())


            D = dist.cdist(np.array([detection_object[-1].centroid for detection_object in self.objects.values()]),
                           np.array([detection_object.centroid for detection_object in detection_objects]))

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                if D[row, col] > self.maxDistance:
                    continue

                detection_objects[col].count, detection_objects[col].unique_id = self.objects[objectID][
                                                                                     -1].count + 1, objectID
                self.objects[objectID].append(detection_objects[col])
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(detection_objects[col])

        return detection_objects
