"""Tests for the VLM adapter layer (visiongraph/adapters.py) - proves
VisionNode can take a real/external backend via an adapter object with zero
changes to state.py, nodes.py, or graph.py.
"""

import json

import pytest

from visiongraph import VisionGraph, VisionNode
from visiongraph.adapters import FileBackedVLMAdapter, PlaceholderVLMAdapter


class TestPlaceholderVLMAdapter:
    def test_runs_through_vision_node_with_no_core_changes(self):
        graph = VisionGraph(name="adapter-smoke-test")
        graph.add_node(
            "vision",
            VisionNode(prompt="What do you see?", infer_fn=PlaceholderVLMAdapter()),
        )
        graph = graph.compile()

        final = graph.run(image="fake.png", prompt="What do you see?")

        assert "placeholder adapter" in final.visual_state.scene_description
        assert final.history[-1].node_name == "VisionNode"


class TestFileBackedVLMAdapter:
    def test_loads_a_colab_produced_result_into_visual_state(self, tmp_path):
        result_file = tmp_path / "phi_result.json"
        result_file.write_text(
            json.dumps(
                {
                    "scene": "A lion resting in dry grass, looking toward the camera.",
                    "objects": [],
                    "text": [],
                }
            ),
            encoding="utf-8",
        )

        graph = VisionGraph(name="phi-file-bridge-test")
        graph.add_node(
            "vision",
            VisionNode(
                prompt="Identify the objects in this image.",
                infer_fn=FileBackedVLMAdapter(str(result_file)),
            ),
        )
        graph = graph.compile()

        final = graph.run(image="Lion.jfif", prompt="Identify the objects in this image.")

        assert final.visual_state.scene_description == (
            "A lion resting in dry grass, looking toward the camera."
        )

    def test_raises_a_clear_error_when_result_file_is_missing(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.json"
        adapter = FileBackedVLMAdapter(str(missing_path))

        with pytest.raises(FileNotFoundError):
            adapter(image="Lion.jfif", prompt="p")
