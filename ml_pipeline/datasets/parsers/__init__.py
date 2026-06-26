"""Dataset parsers package."""

from ml_pipeline.datasets.parsers.base import BaseDatasetParser
from ml_pipeline.datasets.parsers.daic import DAICParser
from ml_pipeline.datasets.parsers.dvlog import DVLOGParser

PARSER_REGISTRY: dict[str, type[BaseDatasetParser]] = {
    "daic_woz": DAICParser,
    "dvlog": DVLOGParser,
}


def get_parser(name: str, raw_root=None) -> BaseDatasetParser:
    if name not in PARSER_REGISTRY:
        raise ValueError(f"Unknown parser: {name}. Available: {list(PARSER_REGISTRY)}")
    return PARSER_REGISTRY[name](raw_root)


__all__ = ["BaseDatasetParser", "DAICParser", "DVLOGParser", "get_parser", "PARSER_REGISTRY"]
