import numpy as np
from modules.stream.source_loader import OpencvReader


class SourceAnalyzer:
    def __init__(self, reader: OpencvReader):
        self.reader = reader

    def run(self):
        while True:
            frames, ids = self.reader.get_frames()
            for frame, _id in zip(frames, ids):
                if isinstance(frame, np.ndarray):
                    self.reader.show_frames(frame, _id)
