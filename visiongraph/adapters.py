"""VLM adapters: pluggable backends for VisionNode.infer_fn.

Kept separate from state.py/nodes.py/graph.py (and not imported by
__init__.py) so the core graph never knows or depends on where a VLM
actually runs. An adapter is just any callable matching VisionNode's
infer_fn contract: adapter(image, prompt) -> dict(scene=..., objects=...,
text=...). See nodes.py::VisionNode._to_visual_state for that contract.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class PlaceholderVLMAdapter:
    """No backend wired up yet. Proves VisionNode can accept an adapter
    object (not just a bare function) with zero changes to the core graph.
    """

    def __call__(self, image: Any, prompt: str) -> Dict[str, Any]:
        return {
            "scene": f"[placeholder adapter] no VLM connected yet (prompt: {prompt!r})",
            "objects": [],
            "text": [],
        }


class FileBackedVLMAdapter:
    """Reads a pre-computed VLM result from a JSON file instead of calling a
    model directly.

    This is the bridge to Colab: run Phi-3.5-Vision there, write its output
    to a JSON file in this shape, copy that file next to the local repo,
    and this adapter feeds the real result into VisionNode untouched.

    Expected file contents: {"scene": str, "objects": [...], "text": [...]}
    """

    def __init__(self, result_path: str):
        self.result_path = Path(result_path)

    def __call__(self, image: Any, prompt: str) -> Dict[str, Any]:
        if not self.result_path.exists():
            raise FileNotFoundError(
                f"No VLM result at {self.result_path}. "
                "Run the Colab cell that writes this file first, then copy it here."
            )
        with open(self.result_path, "r", encoding="utf-8") as f:
            return json.load(f)
