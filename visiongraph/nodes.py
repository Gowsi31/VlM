"""Node base class and the V0.1 node types: VisionNode, ToolNode, DecisionNode,
ReasoningNode, ActionNode, HumanNode.

Nodes that need an external backend (a VLM adapter, an LLM call, a
click/type executor, a human prompt) take it as an injected callable or
adapter (`adapter`, `reasoning_fn`, `action_fn`, `tool_fn`, `approve_fn`).
Without one, the node falls back to a clearly-labeled stub so a graph can
be wired and run end-to-end before any real backend is plugged in.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from PIL import Image

from visiongraph.adapters import VLMAdapter, VLMResponse, VLMResponseError, extract_structured_json
from visiongraph.diff import diff_visual_states
from visiongraph.state import DetectedObject, GraphState, VisualState

_VERIFICATION_BASELINE_KEY = "_verification_baseline"
_VERIFICATION_DIFF_KEY = "verification_diff"


def to_pil_image(image: Any) -> Image.Image:
    """Normalizes whatever GraphState.image holds into a PIL.Image.Image.
    This is the one place that happens - adapters only ever see a real
    PIL image, never a path/bytes, so every backend shares the same input
    contract."""
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (str, Path)):
        return Image.open(image)
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image))
    raise TypeError(f"Cannot convert {type(image)} to PIL.Image.Image")


def default_free_text_parse_fn(response: VLMResponse) -> VisualState:
    """Used when VisionNode(structured_output=False) (the default). Never
    inspects response.text for embedded JSON - a stray '{' in ordinary
    prose (a quoted sign, a code snippet the model described, ...) is not
    evidence the model was attempting structured output, so it must not be
    treated as one. Whether JSON parsing is even attempted is the caller's
    explicit choice (structured_output), not something inferred from the
    response body."""
    return VisualState(scene_description=response.text)


def default_structured_parse_fn(response: VLMResponse) -> VisualState:
    """Used when VisionNode(structured_output=True) - the caller has
    explicitly said this prompt asks for structured output. Uses
    response.structured if the adapter already provided validated
    structured output (Mode C); otherwise best-effort extracts JSON
    embedded in the free-text response (Mode B); otherwise falls back to
    free text alone (the model was asked for JSON but didn't produce any)."""
    structured = response.structured
    if structured is None:
        structured = extract_structured_json(response.text)

    if structured is None:
        return VisualState(scene_description=response.text)

    if not isinstance(structured.get("objects"), list):
        raise VLMResponseError(f"Structured output missing a valid 'objects' list: {structured!r}")

    objects = [
        DetectedObject(**obj) if isinstance(obj, dict) else obj for obj in structured["objects"]
    ]
    return VisualState(
        objects=objects,
        scene_description=structured.get("scene", response.text),
        text_content=structured.get("text", []),
        confidence=structured.get("confidence", 0.0),
    )


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
    """Visual perception: calls a VLMAdapter and produces a VisualState.

    image normalization (to_pil_image) and VLMResponse -> VisualState
    conversion (parse_fn) both happen here - the adapter only ever talks to
    the provider and returns a VLMResponse.
    """

    def __init__(
        self,
        prompt: str,
        adapter: Optional[VLMAdapter] = None,
        structured_output: bool = False,
        parse_fn: Optional[Callable[[VLMResponse], VisualState]] = None,
        vision_tools: Optional[List[str]] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name, {"prompt": prompt, "structured_output": structured_output})
        self.prompt = prompt
        self.adapter = adapter
        self.structured_output = structured_output
        default_fn = default_structured_parse_fn if structured_output else default_free_text_parse_fn
        self.parse_fn = parse_fn or default_fn
        self.vision_tools = vision_tools or []

    def execute(self, state: GraphState) -> GraphState:
        if self.adapter is not None:
            image = to_pil_image(state.image)
            response = self.adapter.generate(image, self.prompt)
            if not isinstance(response, VLMResponse):
                raise TypeError(
                    f"VisionNode adapter must return a VLMResponse, got {type(response)}"
                )
        else:
            response = VLMResponse(text="[no VLM backend configured]")

        visual_state = self.parse_fn(response)
        return state.update(visual_state=visual_state).log_step(
            self.name, summary=visual_state.scene_description
        )


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


class VerificationNode(Node):
    """Compares the current VisualState against the previous one it saw and
    records a VisualStateDiff. No pluggable backend - this is pure local
    computation, not something that talks to the outside world.

    Drop the same VerificationNode instance into the graph after every
    observation in a loop: the first time there's nothing to compare
    against yet (verification_diff is None, the baseline is just recorded);
    every time after that it produces a real diff against what it saw last.
    """

    def execute(self, state: GraphState) -> GraphState:
        previous = state.memory.get(_VERIFICATION_BASELINE_KEY)
        current = state.visual_state

        diff = diff_visual_states(previous, current) if previous is not None else None

        memory = {**state.memory, _VERIFICATION_BASELINE_KEY: current, _VERIFICATION_DIFF_KEY: diff}
        if diff is None:
            summary = "baseline captured, nothing to compare yet"
        else:
            summary = (
                f"added={len(diff.added)} removed={len(diff.removed)} "
                f"state_changes={len(diff.state_changes)}"
            )
        return state.update(memory=memory).log_step(self.name, summary=summary)
