"""VLM adapters: pluggable backends for VisionNode.

Kept separate from state.py/nodes.py/graph.py (and not imported by
__init__.py) so the core graph never knows or depends on where or how a
VLM actually runs.

Contract: a VLMAdapter is any object exposing

    generate(image: PIL.Image.Image, prompt: str, **kwargs) -> VLMResponse

No inheritance required (see the VLMAdapter Protocol below) - PhiAdapter,
an OpenAI adapter, a mock, or a file-backed stand-in all just need to match
this shape. VLMResponse is the provider/model boundary object (what a
backend actually said); VisionGraph's domain model (VisualState) is built
from it separately, by VisionNode's parse_fn (see nodes.py).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Protocol, runtime_checkable

from PIL import Image


# ---------------------------------------------------------------------------
# Provider-boundary response and errors
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VLMResponse:
    """What a VLM adapter returns. Not VisionGraph's domain model - just
    what the provider said."""

    text: str
    structured: Optional[Dict[str, Any]] = None
    raw: Any = None


class VLMError(Exception):
    """Base class for all VLM adapter/parsing errors."""


class VLMTimeoutError(VLMError):
    """The provider call took too long."""


class VLMRateLimitError(VLMError):
    """The provider rejected the call due to rate limiting."""


class VLMResponseError(VLMError):
    """The model responded, but its output couldn't be used: malformed or
    incomplete structured output, an unparseable result, etc. Raised
    instead of silently building a VisualState from broken data."""


class VLMConnectionError(VLMError):
    """Could not reach the provider at all."""


@runtime_checkable
class VLMAdapter(Protocol):
    """Behavioral contract, not a base class. Any object with a matching
    generate() satisfies this - no `class PhiAdapter(VLMAdapter)` needed."""

    def generate(self, image: Image.Image, prompt: str, **kwargs: Any) -> VLMResponse:
        ...


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


class PlaceholderVLMAdapter:
    """No backend wired up yet. Proves VisionNode can accept an adapter
    object with zero changes to the core graph."""

    def generate(self, image: Image.Image, prompt: str, **kwargs: Any) -> VLMResponse:
        return VLMResponse(text=f"[placeholder adapter] no VLM connected yet (prompt: {prompt!r})")


class FileBackedVLMAdapter:
    """Reads a pre-computed VLMResponse from a JSON file instead of calling
    a model directly - the manual bridge to Colab: run a real VLM there,
    write its output as {"text": str, "structured": dict|null}, copy that
    file next to the local repo, and this adapter feeds it into VisionNode
    untouched.
    """

    def __init__(self, result_path: str):
        self.result_path = Path(result_path)

    def generate(self, image: Image.Image, prompt: str, **kwargs: Any) -> VLMResponse:
        if not self.result_path.exists():
            raise FileNotFoundError(
                f"No VLM result at {self.result_path}. "
                "Run the Colab cell that writes this file first, then copy it here."
            )
        with open(self.result_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return VLMResponse(text=data.get("text", ""), structured=data.get("structured"), raw=data)


class CallableVLMAdapter:
    """Wraps a plain function as a VLMAdapter - for tests and quick scripts
    that don't need a full adapter class."""

    def __init__(self, fn: Callable[[Image.Image, str], VLMResponse]):
        self._fn = fn

    def generate(self, image: Image.Image, prompt: str, **kwargs: Any) -> VLMResponse:
        return self._fn(image, prompt)


# ---------------------------------------------------------------------------
# Defensive structured-output parsing
#
# Extraction order (bare JSON -> markdown-fenced -> brace-matched) mirrors
# what was empirically tested against real Phi-3.5-Vision output in Colab.
# A UI screenshot caused Phi to start a JSON object, generate repetitive
# objects, and never close the brace even at 600 tokens - that failure mode
# is treated as a real error (VLMResponseError), not a silent fallback to
# free text, so a truncated/corrupted response never quietly becomes an
# empty-objects VisualState.
# ---------------------------------------------------------------------------


def extract_structured_json(text: str) -> Optional[Dict[str, Any]]:
    """Best-effort JSON extraction from free text.

    Returns None if the text has no '{' at all (the model wasn't attempting
    structured output - plain prose is expected and fine). Raises
    VLMResponseError if a '{' is present but no candidate parses (the model
    attempted structured output and failed).
    """
    if "{" not in text:
        return None

    candidates = [text.strip()]

    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        candidates.append(fence_match.group(1))

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    raise VLMResponseError(
        "Model output looked like an attempted structured response but no valid "
        f"JSON could be extracted: {text[:200]!r}"
    )
