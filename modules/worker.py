from .stream.source_loader import OpencvReader
from .stream.source_analyzer import SourceAnalyzer


def run(source_list: list) -> None:
    reader = OpencvReader(source_list).start()

    source_run = SourceAnalyzer(reader)
