import argparse
from typing import List
from objects.source import Source
from .type_func import str_int_arg


def Parsing() -> List:
    parser = argparse.ArgumentParser(description='Process some sources')
    parser.add_argument('sources', type=str_int_arg, nargs='+', help='sources for process')

    source_list = [Source(rtsp=parser.parse_args().sources[i]) for i in range(len(parser.parse_args().sources))]

    return source_list
