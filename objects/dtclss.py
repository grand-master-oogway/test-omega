import numpy as np
from typing import Union
from dataclasses import dataclass

@dataclass
class Source:
    rtsp: Union[str, int]
    frame: np.ndarray = 0