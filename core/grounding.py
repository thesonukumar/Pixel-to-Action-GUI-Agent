"""
core/grounding.py
=================
OmniParser v2 grounding module — screen element detection and bounding box registry.

WHY THIS FILE EXISTS:
    The Ground phase (Phase 2 of the cognitive loop) converts a raw screenshot
    into a structured bounding box registry that:
      1. Detects all interactable UI elements (buttons, inputs, dropdowns, links)
      2. Assigns each a unique numeric label [1], [2], [3]...
      3. Draws colored bounding boxes on the screenshot for Gemini to read
      4. Returns a registry mapping element_id → bounding box data

    This module operates entirely locally using the OmniParser v2 YOLO model,
    offloading no visual data to the cloud during detection.

    LETTERBOX NOTE:
        YOLO models resize all inputs to a fixed square (e.g., 640×640) by padding
        with gray borders (letterboxing). The bounding box coordinates OmniParser
        returns are in this padded space, NOT the original image space.
        The actuator (Stage 2 of the coordinate pipeline) reverses this transformation.
        Grounding stores the original image dimensions and model input size in the
        registry so actuator.py can perform the math correctly.

USAGE:
    from core.grounding import GroundingEngine
    engine = GroundingEngine()
    annotated_bytes, registry = engine.run(screenshot_bytes)
    # registry: {1: {"x_center": 240.0, "y_center": 380.0, "width": 120.0, "height": 40.0}, ...}
"""

import io
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Type Aliases
# ---------------------------------------------------------------------------

BoundingBoxEntry = dict[str, float]
"""
A single bounding box entry in the registry.
Keys: x_center, y_center, width, height — all in PADDED model input pixel space.
The actuator must apply letterbox compensation before using these for clicks.
"""

BoundingBoxRegistry = dict[int, BoundingBoxEntry]
"""
Maps element_id (int) → BoundingBoxEntry.
Example: {1: {"x_center": 320.0, "y_center": 240.0, "width": 80.0, "height": 30.0}}
"""

GroundingResult = tuple[bytes, BoundingBoxRegistry]
"""
Return type of GroundingEngine.run():
  - bytes:               PNG-encoded annotated screenshot with bounding box overlays.
  - BoundingBoxRegistry: The element_id → bounding box mapping.
"""


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class GroundingError(Exception):
    """
    WHAT: Raised when the Grounding phase fails (model not loaded, inference error).
    WHY:  Typed exception allows the main loop to handle grounding failures
          distinctly from perception or planning failures.
    """
    pass


# ---------------------------------------------------------------------------
# Color palette for bounding box overlays
# ---------------------------------------------------------------------------

# Bright, high-contrast colors for bounding boxes — cycles for readability
_BOX_COLORS: list[tuple[int, int, int]] = [
    (255,  87,  34),   # Deep Orange
    ( 33, 150, 243),   # Blue
    ( 76, 175,  80),   # Green
    (255, 193,   7),   # Amber
    (156,  39, 176),   # Purple
    (  0, 188, 212),   # Cyan
    (233,  30,  99),   # Pink
    (255, 152,   0),   # Orange
]

_LABEL_FONT_SIZE: int = 14
_BOX_LINE_WIDTH: int = 2


# ---------------------------------------------------------------------------
# GroundingEngine — Main class
# ---------------------------------------------------------------------------

class GroundingEngine:
    """
    WHAT: Runs OmniParser v2 (YOLO-based) on a screenshot to detect UI elements,
          assigns numeric IDs, draws colored bounding box overlays, and returns
          both the annotated image and a structured bounding box registry.

    WHY:  Abstracts the OmniParser model lifecycle — lazy loading on first call,
          error handling, and annotated output generation — behind a clean interface.

    Attributes:
        _model_path:       Path to the OmniParser v2 .pt weights file.
        _model_input_size: The square size YOLO resizes input to (default 640).
        _model:            The loaded YOLO model (None until first call to run()).
    """

    def __init__(self) -> None:
        """
        WHAT: Configures the GroundingEngine using settings.py values.
        WHY:  Separating config from loading enables lazy initialization —
              the model is only loaded when run() is first called.
        """
        self._model_path: str = settings.OMNIPARSER_MODEL_PATH
        self._model_input_size: int = settings.OMNIPARSER_MODEL_INPUT_SIZE
        self._model = None  # Lazy-loaded on first run()

    def _load_model(self) -> None:
        """
        WHAT: Loads the OmniParser v2 YOLO model from disk into memory.
        WHY:  Lazy loading avoids the ~2-5 second startup cost on every agent
              restart if the model is only needed for grounding.
              Loads only once — subsequent calls to run() reuse the cached model.

        Raises:
            GroundingError: If the model file is not found or fails to load.
        """
        if self._model is not None:
            return  # Already loaded

        if not Path(self._model_path).exists():
            raise GroundingError(
                f"OmniParser model not found at: {self._model_path}\n"
                f"Please download it and place it at the configured OMNIPARSER_MODEL_PATH.\n"
                f"See models/README.txt for instructions."
            )

        try:
            from ultralytics import YOLO  # Import here to avoid startup cost
            self._model = YOLO(self._model_path)
            log.ground(f"OmniParser v2 model loaded from: {self._model_path}")
        except Exception as exc:
            raise GroundingError(f"Failed to load OmniParser model: {exc}") from exc

    def run(self, screenshot_bytes: bytes) -> GroundingResult:
        """
        WHAT: Runs OmniParser v2 inference on the screenshot, draws numeric-labeled
              bounding boxes, and returns the annotated image + element registry.

        WHY:  This is the core of the Ground phase. The annotated image is what
              Gemini reads — it must show clear, numbered bounding boxes over every
              interactable element. The registry maps those numbers to pixel positions
              so the actuator can translate VLM decisions to clicks.

        LETTERBOX METADATA:
            The registry includes the original image dimensions (img_width, img_height)
            and the model_input_size. The actuator MUST use these to reverse the
            letterbox transformation (Stage 2 of the coordinate pipeline).

        Args:
            screenshot_bytes: Raw PNG screenshot bytes from PerceptionEngine.capture().

        Returns:
            GroundingResult: (annotated_png_bytes, bounding_box_registry)
                - annotated_png_bytes: PNG bytes of the screenshot with overlaid boxes.
                - bounding_box_registry: dict mapping element_id → bounding box data.

        Raises:
            GroundingError: If inference fails or the screenshot cannot be decoded.
        """
        # Lazy-load the model on first invocation
        self._load_model()

        # --- Decode screenshot bytes → PIL Image ---
        try:
            image: Image.Image = Image.open(io.BytesIO(screenshot_bytes)).convert("RGB")
        except Exception as exc:
            raise GroundingError(f"Failed to decode screenshot bytes: {exc}") from exc

        img_width, img_height = image.size
        log.ground(f"Input image: {img_width}×{img_height} px")

        # --- Run YOLO inference ---
        try:
            # OmniParser processes at model_input_size (e.g. 640×640) internally
            results = self._model(
                source=np.array(image),
                imgsz=self._model_input_size,
                conf=0.25,  # Confidence threshold — lower catches more elements
                verbose=False,
            )
        except Exception as exc:
            raise GroundingError(f"OmniParser inference failed: {exc}") from exc

        # --- Parse detections and build registry ---
        registry: BoundingBoxRegistry = {}
        draw = ImageDraw.Draw(image)

        # Attempt to load a font; fall back to PIL default if not available
        try:
            font = ImageFont.truetype("arial.ttf", _LABEL_FONT_SIZE)
        except OSError:
            font = ImageFont.load_default()

        element_id: int = 1

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # YOLO returns xyxy coordinates in the ORIGINAL image space
                # (ultralytics un-letterboxes automatically via its postprocess)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])

                # Compute center and dimensions in original image space
                x_center = (x1 + x2) / 2.0
                y_center = (y1 + y2) / 2.0
                width    = x2 - x1
                height   = y2 - y1

                # Store in registry — actuator will read these
                registry[element_id] = {
                    "x_center":    x_center,
                    "y_center":    y_center,
                    "width":       width,
                    "height":      height,
                    "confidence":  conf,
                    # Metadata for coordinate pipeline (Stage 2 in actuator.py)
                    "img_width":   float(img_width),
                    "img_height":  float(img_height),
                    "model_input_size": float(self._model_input_size),
                }

                # Pick a color from the palette (cycle through)
                color = _BOX_COLORS[(element_id - 1) % len(_BOX_COLORS)]

                # Draw bounding box
                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline=color,
                    width=_BOX_LINE_WIDTH,
                )

                # Draw label badge background
                label_text = f"[{element_id}]"
                bbox_text = draw.textbbox((x1, y1 - _LABEL_FONT_SIZE - 4), label_text, font=font)
                draw.rectangle(
                    [bbox_text[0] - 2, bbox_text[1] - 2, bbox_text[2] + 2, bbox_text[3] + 2],
                    fill=color,
                )

                # Draw label text in white for contrast
                draw.text(
                    (x1, y1 - _LABEL_FONT_SIZE - 4),
                    label_text,
                    fill=(255, 255, 255),
                    font=font,
                )

                element_id += 1

        log.ground(
            f"Detected {len(registry)} interactable elements. "
            f"Registry IDs: {list(registry.keys())}"
        )

        # --- Encode annotated image back to PNG bytes ---
        output_buffer = io.BytesIO()
        image.save(output_buffer, format="PNG")
        annotated_bytes = output_buffer.getvalue()

        return annotated_bytes, registry

    def get_element_center(
        self,
        registry: BoundingBoxRegistry,
        element_id: int,
    ) -> tuple[float, float]:
        """
        WHAT: Returns the (x_center, y_center) of an element from the registry.
        WHY:  Convenience helper for the actuator — avoids raw dict access.

        Args:
            registry:   The bounding box registry returned by run().
            element_id: The numeric label assigned during grounding.

        Returns:
            tuple[float, float]: (x_center, y_center) in original image pixel space.

        Raises:
            GroundingError: If element_id is not in the registry.
        """
        if element_id not in registry:
            raise GroundingError(
                f"Element ID {element_id} not found in registry. "
                f"Available IDs: {list(registry.keys())}"
            )
        entry = registry[element_id]
        return entry["x_center"], entry["y_center"]
