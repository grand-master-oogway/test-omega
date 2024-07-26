from typing import Union, List
from dataclasses import dataclass


@dataclass
class DetectedObject:
    bbox: List
    class_name: str
    conf: float
    source_id: int
    """fields that will be filling in tracker"""
    centroid: List
    unique_id: Union[None, str] = None
    count: Union[None, int] = None
