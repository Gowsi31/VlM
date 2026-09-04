"""Node base class and the V0.1 node types: VisionNode, ToolNode, DecisionNode,
ReasoningNode, ActionNode, HumanNode.

Nodes that need an external backend (a VLM call, an LLM call, a click/type
executor, a human prompt) take it as an injected callable (`infer_fn`,
`reasoning_fn`, `action_fn`, `tool_fn`, `approve_fn`). Without one, the node
falls back to a clearly-labeled stub so a graph can be wired and run
end-to-end before any real backend is plugged in.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from visiongraph.state import DetectedObject, GraphState, VisualState


class Node:
    """Base class for all graph nodes."""

    def __init__(self, name: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        self.name = name or self.__class__.__name__
        self.config = config or {}

    def execute(self, state: GraphState) -> GraphState:
        raise NotImplementedError(f"{self.__class__.__name__} must implement execute()")

    def validate(self) -> bool:
        return bool(self.name)

    def get_schema(self) -> Dict[str, Any]:
        return {"name": self.name, "type": self.__class__.__name__, "config": self.config}


class VisionNode(Node):
    """Visual perception: calls a VLM (via infer_fn) and produces a VisualState."""

    def __init__(
        self,
        prompt: str,
        output_format: str = "json",
        vision_tools: Optional[List[str]] = None,
        infer_fn: Optional[Callable[[Any, str], Any]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name, {"prompt": prompt, "output_format": output_format})
        self.prompt = prompt
        self.output_format = output_format
        self.vision_tools = vision_tools or []
        self.infer_fn = infer_fn

    def execute(self, state: GraphState) -> GraphState:
        if self.infer_fn is not None:
            result = self.infer_fn(state.image, self.prompt)
        else:
            result = VisualState(
                scene_description="[no VLM backend configured]",
                confidence=0.0,
            )

        visual_state = self._to_visual_state(result)
        return state.update(visual_state=visual_state).log_step(
            self.name, summary=visual_state.scene_description
        )

    @staticmethod
    def _to_visual_state(result: Any) -> VisualState:
        if isinstance(result, VisualState):
            return result
        if isinstance(result, dict):
            objects = [
                DetectedObject(**obj) if isinstance(obj, dict) else obj
                for obj in result.get("objects", [])
            ]
            return VisualState(
                objects=objects,
                scene_description=result.get("scene_description", result.get("scene", "")),
                text_content=result.get("text_content", result.get("text", [])),
                confidence=result.get("confidence", 0.0),
            )
        raise TypeError(f"VisionNode infer_fn must return a VisualState or dict, got {type(result)}")


class ToolNode(Node):
    """Calls an external tool (search, API, database, browser, ...) via tool_fn."""

    def __init__(
        self,
        tool_type: str,
        tool_config: Optional[Dict[str, Any]] = None,
        tool_fn: Optional[Callable[[GraphState], Any]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name, {"tool_type": tool_type, "tool_config": tool_config or {}})
        self.tool_type = tool_type
        self.tool_config = tool_config or {}
        self.tool_fn = tool_fn

    def execute(self, state: GraphState) -> GraphState:
        if self.tool_fn is not None:
            result = self.tool_fn(state)
        else:
            result = f"[no tool backend configured for '{self.tool_type}']"

        memory = {**state.memory, f"tool:{self.tool_type}": result}
        return state.update(memory=memory).log_step(self.name, summary=str(result))


class DecisionNode(Node):
    """Branch logic: evaluates `logic(state)` and stores the chosen next node name."""

    def __init__(self, logic: Callable[[GraphState], str], name: Optional[str] = None):
        super().__init__(name)
        self.logic = logic

    def execute(self, state: GraphState) -> GraphState:
        next_node = self.logic(state)
        return state.update(next_node=next_node).log_step(self.name, summary=f"-> {next_node}")


class ReasoningNode(Node):
    """LLM reasoning over the current state, via reasoning_fn(state, prompt) -> (text, confidence)."""

    def __init__(
        self,
        prompt: str,
        reasoning_fn: Optional[Callable[[GraphState, str], Any]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name, {"prompt": prompt})
        self.prompt = prompt
        self.reasoning_fn = reasoning_fn

    def execute(self, state: GraphState) -> GraphState:
        if self.reasoning_fn is not None:
            reasoning, confidence = self.reasoning_fn(state, self.prompt)
        else:
            reasoning, confidence = "[no reasoning backend configured]", 0.0

        return state.update(reasoning=reasoning, confidence=confidence).log_step(
            self.name, summary=reasoning
        )


class ActionNode(Node):
    """Executes an action (click, type, scroll, wait, ...) via action_fn(state, action_type, config)."""

    def __init__(
        self,
        action_type: str,
        config: Optional[Dict[str, Any]] = None,
        action_fn: Optional[Callable[[GraphState, str, Dict[str, Any]], Any]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name, config)
        self.action_type = action_type
        self.action_fn = action_fn

    def execute(self, state: GraphState) -> GraphState:
        if self.action_fn is not None:
            result = self.action_fn(state, self.action_type, self.config)
        else:
            result = {"status": "stub", "note": "[no action backend configured]"}

        record = {"action_type": self.action_type, "config": self.config, "result": result}
        return state.update(actions_taken=[*state.actions_taken, record]).log_step(
            self.name, summary=f"{self.action_type} -> {result}"
        )


class HumanNode(Node):
    """Requests human approval via approve_fn(state) -> bool. Defaults to auto-approve."""

    def __init__(
        self,
        prompt: str = "Approve?",
        approve_fn: Optional[Callable[[GraphState], bool]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name, {"prompt": prompt})
        self.prompt = prompt
        self.approve_fn = approve_fn

    def execute(self, state: GraphState) -> GraphState:
        if self.approve_fn is not None:
            approved = self.approve_fn(state)
        else:
            approved = True  # [no human backend configured] - auto-approve

        memory = {**state.memory, "human_approved": approved}
        return state.update(memory=memory).log_step(self.name, summary=f"approved={approved}")
