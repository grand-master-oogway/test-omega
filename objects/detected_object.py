from typing import Union, List
from dataclasses import dataclass


@dataclass
class DetectedObject:
    """
    Dataclass for detected objects.
    """
    bbox: List
    class_name: str
    conf: float
    source_id: int
    centroid: List
    """fields that will be filling in tracker"""
    unique_id: Union[None, str] = None
    count: Union[None, int] = None
