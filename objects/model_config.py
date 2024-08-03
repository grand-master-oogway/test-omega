from typing import Union, Tuple
from dataclasses import dataclass
@dataclass
class ModelConfig:
    """
    Dataclass for model configure
    """
    weights: str
    data: str
    imgsz: Tuple[int, int]
    conf_thres: float
    iou_thres: float
    max_det: int
    classes: Union[None, int]  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms: bool
    hide_labels: bool
    hide_conf: bool
    half: bool
    dnn: bool
