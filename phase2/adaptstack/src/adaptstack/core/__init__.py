"""Core contracts used by all AdaptStack layers."""

from .immutable import freeze_mapping, freeze_value, plain_value
from .registry import StageRegistry
from .types import ArtifactRef, PipelineStage, RunContext, StageResult

__all__ = [
    "ArtifactRef",
    "PipelineStage",
    "RunContext",
    "StageRegistry",
    "StageResult",
    "freeze_mapping",
    "freeze_value",
    "plain_value",
]
