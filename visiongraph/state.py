"""State objects that flow through a VisionGraph: DetectedObject, VisualState, GraphState."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class DetectedObject:
    """A single object/element found in an observation."""

    type: str
    label: str
    confidence: float = 0.0
    description: str = ""
    bbox: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    state: Optional[str] = None  # e.g. "selected" / "unselected" - element state, not object identity


@dataclass(frozen=True)
class VisualState:
    """Structured representation of what the VLM perceived in one observation."""

    objects: List[DetectedObject] = field(default_factory=list)
    scene_description: str = ""
    text_content: List[str] = field(default_factory=list)
    confidence: float = 0.0
    timestamp: float = field(default_factory=time.time)


@dataclass(frozen=True)
class StepRecord:
    """A record of one node execution, kept in GraphState.history for debugging/replay."""

    node_name: str
    timestamp: float = field(default_factory=time.time)
    summary: str = ""


@dataclass(frozen=True)
class GraphState:
    """Immutable state passed between nodes. Use .update(**kwargs) to derive a new state."""

    # Input
    image: Any = None
    prompt: str = ""

    # Visual understanding
    visual_state: Optional[VisualState] = None

    # Reasoning
    reasoning: str = ""
    confidence: float = 0.0

    # Execution / control flow
    current_node: Optional[str] = None
    next_node: Optional[str] = None
    step_count: int = 0
    history: List[StepRecord] = field(default_factory=list)

    # Actions
    actions_taken: List[Dict[str, Any]] = field(default_factory=list)

    # Memory (V0.1: plain dict, no persistence across runs)
    memory: Dict[str, Any] = field(default_factory=dict)

    def update(self, **kwargs: Any) -> "GraphState":
        return replace(self, **kwargs)

    def log_step(self, node_name: str, summary: str = "") -> "GraphState":
        record = StepRecord(node_name=node_name, summary=summary)
        return self.update(
            current_node=node_name,
            step_count=self.step_count + 1,
            history=[*self.history, record],
        )
