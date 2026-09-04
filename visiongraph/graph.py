"""VisionGraph: wires nodes into a graph and runs the
observe -> perceive -> reason -> act -> verify loop (V0.1 scope).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

from visiongraph.nodes import Node
from visiongraph.state import GraphState

Condition = Callable[[GraphState], bool]


class VisionGraph:
    def __init__(
        self,
        model: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.name = name or "visiongraph"
        self.config = config or {}
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Tuple[str, str, Optional[Condition]]] = []
        self.entry_point: Optional[str] = None

    def add_node(self, name: str, node: Node) -> "VisionGraph":
        self.nodes[name] = node
        if self.entry_point is None:
            self.entry_point = name
        return self

    def set_entry_point(self, name: str) -> "VisionGraph":
        self.entry_point = name
        return self

    def connect(self, source: str, target: str, condition: Optional[Condition] = None) -> "VisionGraph":
        self.edges.append((source, target, condition))
        return self

    def validate(self) -> bool:
        if not self.nodes or self.entry_point is None:
            return False
        for source, target, _ in self.edges:
            if source not in self.nodes or target not in self.nodes:
                return False
        return True

    def compile(self, mode: str = "single") -> "VisionGraph":
        if not self.validate():
            raise ValueError(
                "Invalid graph: entry point unset, or an edge references an unknown node."
            )
        self.config["mode"] = mode
        return self

    def _next_node(self, current: str, state: GraphState) -> Optional[str]:
        if state.next_node is not None:
            return state.next_node

        fallback = None
        for source, target, condition in self.edges:
            if source != current:
                continue
            if condition is None:
                fallback = target
            elif condition(state):
                return target
        return fallback

    def run(
        self,
        image: Any = None,
        prompt: str = "",
        max_steps: int = 10,
        **kwargs: Any,
    ) -> GraphState:
        if not self.validate():
            raise ValueError(
                "Invalid graph: entry point unset, or an edge references an unknown node."
            )

        state = GraphState(image=image, prompt=prompt, **kwargs)
        current = self.entry_point

        while current is not None and state.step_count < max_steps:
            node = self.nodes[current]
            state = node.execute(state)
            next_node = self._next_node(current, state)
            state = state.update(next_node=None)
            current = next_node

        return state
