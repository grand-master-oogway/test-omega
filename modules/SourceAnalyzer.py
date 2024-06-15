import numpy as np
from .SourceLoader import OpencvReader

class SourceAnalyzer():
    def __init__(self, reader: OpencvReader):
        while True:
            frames, ids = reader.get_frames()
            for frame, _id in zip(frames, ids):
                if isinstance(frame, np.ndarray):
                    reader.show_frames(frame, _id)
