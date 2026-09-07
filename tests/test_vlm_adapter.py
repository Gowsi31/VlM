"""Tests for the VLM adapter layer (visiongraph/adapters.py) - proves
VisionNode can take a real/external backend via an adapter object with zero
changes to state.py, nodes.py, or graph.py.
"""

import json

import pytest
from PIL import Image

from visiongraph import VisionGraph, VisionNode
from visiongraph.adapters import (
    FileBackedVLMAdapter,
    PlaceholderVLMAdapter,
    VLMResponse,
    VLMResponseError,
    extract_structured_json,
)

A_PIXEL = Image.new("RGB", (2, 2))


class TestPlaceholderVLMAdapter:
    def test_runs_through_vision_node_with_no_core_changes(self):
        graph = VisionGraph(name="adapter-smoke-test")
        graph.add_node(
            "vision",
            VisionNode(prompt="What do you see?", adapter=PlaceholderVLMAdapter()),
        )
        graph = graph.compile()

        final = graph.run(image=A_PIXEL, prompt="What do you see?")

        assert "placeholder adapter" in final.visual_state.scene_description
        assert final.history[-1].node_name == "VisionNode"

    def test_generate_returns_a_vlm_response(self):
        response = PlaceholderVLMAdapter().generate(A_PIXEL, "What do you see?")

        assert isinstance(response, VLMResponse)
        assert response.structured is None


class TestFileBackedVLMAdapter:
    def test_loads_a_colab_produced_result_into_visual_state(self, tmp_path):
        result_file = tmp_path / "phi_result.json"
        result_file.write_text(
            json.dumps(
                {
                    "text": "A lion resting in dry grass, looking toward the camera.",
                    "structured": None,
                }
            ),
            encoding="utf-8",
        )

        graph = VisionGraph(name="phi-file-bridge-test")
        graph.add_node(
            "vision",
            VisionNode(
                prompt="Identify the objects in this image.",
                adapter=FileBackedVLMAdapter(str(result_file)),
            ),
        )
        graph = graph.compile()

        final = graph.run(image=A_PIXEL, prompt="Identify the objects in this image.")

        assert final.visual_state.scene_description == (
            "A lion resting in dry grass, looking toward the camera."
        )

    def test_raises_a_clear_error_when_result_file_is_missing(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.json"
        adapter = FileBackedVLMAdapter(str(missing_path))

        with pytest.raises(FileNotFoundError):
            adapter.generate(A_PIXEL, "p")


class TestExtractStructuredJson:
    """Extraction order (bare -> fenced -> brace-matched) and the
    attempted-but-failed error case were both validated against real
    Phi-3.5-Vision output in Colab."""

    def test_returns_none_for_plain_prose_with_no_json_attempt(self):
        assert extract_structured_json("A lion walking across a grassy plain.") is None

    def test_parses_bare_json(self):
        result = extract_structured_json('{"scene": "a lion", "objects": []}')

        assert result == {"scene": "a lion", "objects": []}

    def test_parses_json_wrapped_in_a_markdown_fence(self):
        text = '```json\n{"scene": "a lion", "objects": []}\n```'

        result = extract_structured_json(text)

        assert result == {"scene": "a lion", "objects": []}

    def test_parses_json_preceded_by_prose(self):
        text = 'Here is the result:\n{"scene": "a lion", "objects": []}'

        result = extract_structured_json(text)

        assert result == {"scene": "a lion", "objects": []}

    def test_raises_on_truncated_json(self):
        """The actual failure mode observed with Phi against a UI
        screenshot: it starts a JSON object and never closes it."""
        truncated = '{"scene": "a UI screen", "objects": [{"type": "button", "label": "profile"'

        with pytest.raises(VLMResponseError):
            extract_structured_json(truncated)
