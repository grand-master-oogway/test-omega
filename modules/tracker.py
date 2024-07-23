from __future__ import annotations

import numpy as np
from typing import List, Tuple, Any, Type
from collections import OrderedDict
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, maxDisappeared=30):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.originRects = OrderedDict()
        self.disappeared = OrderedDict()

        self.maxDisappeared = maxDisappeared

    def register(self, centroid, rect) -> None:
        self.originRects[self.nextObjectID] = rect
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID) -> None:
        del self.originRects[objectID]
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects: List[DetectedObjects]) -> Tuple[int, List[int], List[List[int]]]:

        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1

                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            count = len(self.objects)
            ids = list(self.objects.keys())
            centre = list(self.objects.values())
            return count, ids, centre

        inputCentroids = np.zeros((len(rects), 2), dtype="int")

        for i, rect in enumerate(rects):
            cX = rect.centroid[0]
            cY = rect.centroid[1]
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                centroid = inputCentroids[i]
                rect = rects[i]
                self.register(centroid, rect)

        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):

                if row in usedRows or col in usedCols:
                    continue

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.originRects[objectID] = rects[col]
                self.disappeared[objectID] = 0

                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            if D.shape[0] >= D.shape[1]:

                for row in unusedRows:
                    objectID = objectIDs[row]
                    self.disappeared[objectID] += 1

                    if self.disappeared[objectID] > self.maxDisappeared:
                        self.deregister(objectID)

            else:
                for col in unusedCols:
                    centroid = inputCentroids[col]
                    rect = rects[col]
                    self.register(centroid, rect)


        count = len(self.objects)
        ids = list(self.objects.keys())
        centre = list(self.objects.values())
        return count, ids, centre
