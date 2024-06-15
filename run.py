from __future__ import annotations

import argparse
import numpy as np
from objects import Source
from modules import str_int_arg
from modules import OpencvReader



from icecream import ic


def main():
    parser = argparse.ArgumentParser(description='Process some sources')
    parser.add_argument('sources', type=str_int_arg, nargs='+', help='sources for process')

    source_list = [Source(rtsp=parser.parse_args().sources[i]) for i in range(len(parser.parse_args().sources))]

    reader = OpencvReader(source_list).start()

    while True:
        frames, ids = reader.get_frames()
        ic(frames)
        for frame, _id in zip(frames, ids):
            if isinstance(frame, np.ndarray):
                ic(reader.show_frames(frame, _id))
                reader.show_frames(frame, _id)


if __name__ == '__main__':
    main()
