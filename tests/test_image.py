"""Tests for visiongraph/image.py (crop_image) and its use ahead of
VisionNode - cropping happens before the node, never inside it.
"""

import pytest
from PIL import Image

from visiongraph.adapters import CallableVLMAdapter, VLMResponse
from visiongraph.image import crop_image
from visiongraph.nodes import VisionNode
from visiongraph.state import GraphState


class TestCropImage:
    def test_valid_crop_returns_expected_region_and_dimensions(self):
        image = Image.new("RGB", (200, 200))

        cropped = crop_image(image, (10, 10, 60, 90))

        assert isinstance(cropped, Image.Image)
        assert cropped.size == (50, 80)

    def test_crop_matching_real_calendar_region_coordinates(self):
        # The exact crop from the real Colab experiment: a 1500x1000
        # screenshot, cropped down to just the calendar panel.
        screenshot = Image.new("RGB", (1500, 1000))

        cropped = crop_image(screenshot, (695, 715, 1165, 920))

        assert cropped.size == (1165 - 695, 920 - 715)

    def test_non_pil_input_raises_type_error(self):
        with pytest.raises(TypeError):
            crop_image("not_an_image.png", (0, 0, 10, 10))

    def test_bbox_with_wrong_length_raises_value_error(self):
        image = Image.new("RGB", (100, 100))

        with pytest.raises(ValueError):
            crop_image(image, (0, 0, 10))

    def test_bbox_with_non_numeric_coordinates_raises_value_error(self):
        image = Image.new("RGB", (100, 100))

        with pytest.raises(ValueError):
            crop_image(image, ("a", "b", "c", "d"))

    def test_bbox_where_left_is_not_less_than_right_raises_value_error(self):
        image = Image.new("RGB", (100, 100))

        with pytest.raises(ValueError):
            crop_image(image, (60, 10, 10, 90))

    def test_bbox_where_top_is_not_less_than_bottom_raises_value_error(self):
        image = Image.new("RGB", (100, 100))

        with pytest.raises(ValueError):
            crop_image(image, (10, 90, 60, 10))

    def test_bbox_with_negative_coordinate_raises_value_error(self):
        image = Image.new("RGB", (100, 100))

        with pytest.raises(ValueError):
            crop_image(image, (-5, 10, 60, 90))

    def test_bbox_exceeding_image_bounds_raises_value_error(self):
        image = Image.new("RGB", (100, 100))

        with pytest.raises(ValueError):
            crop_image(image, (10, 10, 150, 90))


class TestCropThenVisionNode:
    def test_full_image_crop_region_vision_node_visual_state(self):
        """full image -> crop region -> VisionNode -> VisualState, with
        VisionNode completely unaware cropping happened - it just receives
        an already-cropped PIL image, same as any other image."""
        full_screenshot = Image.new("RGB", (1500, 1000))
        calendar_region = crop_image(full_screenshot, (695, 715, 1165, 920))

        received = {}

        def fake_generate(image, prompt):
            received["size"] = image.size
            return VLMResponse(
                text="calendar view",
                structured={
                    "scene": "calendar view",
                    "objects": [
                        {"type": "calendar", "label": "7", "state": "selected"},
                        {"type": "calendar", "label": "8", "state": "unselected"},
                    ],
                },
            )

        node = VisionNode(
            prompt="What's the calendar state?",
            adapter=CallableVLMAdapter(fake_generate),
            structured_output=True,
        )
        result = node.execute(GraphState(image=calendar_region))

        # Proves the crop happened before VisionNode: the adapter received
        # the small cropped region, not the full 1500x1000 screenshot.
        assert received["size"] == (1165 - 695, 920 - 715)
        assert result.visual_state.objects[0].label == "7"
        assert result.visual_state.objects[0].state == "selected"
        assert result.visual_state.objects[1].label == "8"
        assert result.visual_state.objects[1].state == "unselected"
