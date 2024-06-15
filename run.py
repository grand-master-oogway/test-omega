from __future__ import annotations

from modules import Parsing
from modules import run


def main():
    source_list = Parsing()
    run(source_list)


if __name__ == '__main__':
    main()
