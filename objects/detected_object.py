from typing import Union, List
from dataclasses import dataclass

@dataclass
class DetectedObject:
    bbox: List
    class_name: str
    conf: float
    centroid: List
    source_id: int
