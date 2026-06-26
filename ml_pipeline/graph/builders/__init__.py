from ml_pipeline.graph.builders.base import BaseGraphBuilder
from ml_pipeline.graph.builders.clinical_builder import ClinicalGraphBuilder
from ml_pipeline.graph.builders.neo4j_builder import Neo4jGraphBuilder
from ml_pipeline.graph.builders.research_builder import ResearchGraphBuilder
from ml_pipeline.graph.builders.temporal_builder import TemporalGraphBuilder

BUILDER_REGISTRY = {
    "neo4j": Neo4jGraphBuilder,
    "temporal": TemporalGraphBuilder,
    "clinical": ClinicalGraphBuilder,
    "research": ResearchGraphBuilder,
}


def build_graph_builder(name: str = "temporal", **kwargs) -> BaseGraphBuilder:
    cls = BUILDER_REGISTRY.get(name, TemporalGraphBuilder)
    return cls(**kwargs)
