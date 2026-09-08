"""Image-region preprocessing: crop a focused region out of a larger image
before it reaches VisionNode.

Kept entirely separate from VisionNode - cropping is something a caller
does to an image before handing it to a graph run (e.g. GraphState(image=
crop_image(screenshot, calendar_bbox))), not something VisionNode does
itself. Motivated by a real finding: Phi-3.5-Vision reliably read a small
calendar's selection state from a focused crop but not from the full
1500x1000 screenshot - the issue was model attention/resolution on a small
region, not the VisualState/diff/VerificationNode implementation.
"""

from __future__ import annotations

from typing import Tuple

from PIL import Image


def crop_image(image: Image.Image, bbox: Tuple[int, int, int, int]) -> Image.Image:
    """Crop `image` to `bbox = (left, top, right, bottom)`, PIL's own crop
    semantics. Raises a clear error on invalid input rather than silently
    producing an empty or out-of-bounds crop."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"crop_image expects a PIL.Image.Image, got {type(image)}")

    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        raise ValueError(f"bbox must be a 4-tuple (left, top, right, bottom), got {bbox!r}")

    left, top, right, bottom = bbox
    for value in (left, top, right, bottom):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"bbox coordinates must be numeric, got {bbox!r}")

    if left >= right or top >= bottom:
        raise ValueError(f"bbox must satisfy left < right and top < bottom, got {bbox!r}")

    width, height = image.size
    if left < 0 or top < 0 or right > width or bottom > height:
        raise ValueError(f"bbox {bbox!r} is out of bounds for image of size {image.size}")

    return image.crop((left, top, right, bottom))
