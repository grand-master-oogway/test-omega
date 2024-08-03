import abc
import statistics
import numpy as np
from enum import Enum
from typing import List, Dict

from objects import DetectedObject


class DictKeyNames(Enum):
    CLASS_ID: str = "class_id"


class Handler:
    """
    Class for tracking and handling some objects.
    """

    def __init__(self, max_disappeared: int = 20, max_length_buff: int = 21):
        """
        Constructor for Handler class.

        Args:
            max_disappeared (int): The maximum number of consecutive frames an object can be absent before it
                is considered as disappeared. Default value is 20.
            max_length_buff (int): The maximum length of the buffer that stores object's location history.
                Default value is 21.
        """
        self._max_disappeared: int = max_disappeared
        self._max_length_buff: int = max_length_buff

        self._obj_history: Dict = dict()
        self._disappeared: Dict = dict()
        # self._ids_list: List = []

    def _register(self, obj: DetectedObject) -> None:
        """
        Adds a new  object to the handler system.

        Args:
            obj (some_obj_type): object to be registered.
        """
        self._obj_history[obj.unique_id] = {
            DictKeyNames.CLASS_ID.value: [],
        }
        self._disappeared[obj.unique_id] = 0

    def _deregister(self, objectID: str) -> None:
        """
        Removes object from the handler system.

        Args:
            objectID: The ID of object to be removed.
        """
        del self._disappeared[objectID]
        del self._obj_history[objectID]

    def update(self, some_objects: List[DetectedObject]) -> List[DetectedObject]:
        """
        Updates the location history of existing objects and adds new objects to the handler
        system. Removes object that have disappeared for a long time.

        Args:
            some_objects (List[some_obj_type]): A list of objects to be processed.

        Returns:
            List[some_obj_type]: A list of objects that have reached the maximum length of their history buffer
            and are ready to be outputted.
        """
        output_objects = []

        if len(some_objects) == 0:
            for objectID in list(self._disappeared.keys()):
                self._disappeared[objectID] += 1

                if self._disappeared[objectID] > self._max_disappeared:
                    self._deregister(objectID)

            return output_objects

        if len(self._obj_history) == 0:
            for obj in some_objects:
                self._register(obj)
        else:
            used_ids = []

            for obj in some_objects:
                if obj.unique_id not in used_ids:
                    used_ids.append(obj.unique_id)

                if obj.unique_id not in self._obj_history:
                    self._register(obj)

                elif obj.count < self._max_length_buff:
                    self._disappeared[obj.unique_id] = 0
                    self._obj_history[obj.unique_id][DictKeyNames.CLASS_ID.value].append(obj.class_name)

                elif obj.count == self._max_length_buff:
                    output_objects.append(self._find_best_entry(obj))

            for objectID in set(self._obj_history.keys()) - set(used_ids):
                self._disappeared[objectID] += 1
                if self._disappeared[objectID] > self._max_disappeared:
                    self._deregister(objectID)

        return output_objects

    def _find_best_entry(self, obj: DetectedObject) -> DetectedObject:
        """
        Find the best entry for a given object by calculating the mode of the class_id.

        Args:
            obj (some_obj_type): object.
        Returns:
            obj: object with the best entry.
        """
        obj.class_id = statistics.mode(
            [
                class_id for class_id in
                self._obj_history[obj.unique_id]
                [DictKeyNames.CLASS_ID.value]
            ]
        )
        return obj
