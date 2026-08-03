"""Gemma Expert Atlas: evidence-first MoE routing feasibility analysis."""

from .decision import DEFAULT_GATES, evaluate_decision
from .model import AtlasManifest, RouteRecord

__all__ = ["AtlasManifest", "RouteRecord", "DEFAULT_GATES", "evaluate_decision"]
__version__ = "0.1.0"
