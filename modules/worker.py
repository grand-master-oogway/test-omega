from .SourceLoader import OpencvReader
from .SourceAnalyzer import SourceAnalyzer


def run(source_list: list) -> None:
    reader = OpencvReader(source_list).start()

    source_run = SourceAnalyzer(reader)
