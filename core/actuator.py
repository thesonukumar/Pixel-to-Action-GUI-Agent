"""
core/actuator.py
================
3-Stage Coordinate Transformation Pipeline and Playwright Action Executor.

WHY THIS FILE EXISTS:
    The Act phase (Phase 4 of the cognitive loop) must translate the VLM's abstract
    element_id selection into exact physical OS pixel coordinates, then dispatch the
    correct Playwright command.

    This requires 3 mathematical stages as specified in Section 4 of the master directive:

    Stage 1 — Bounding Box Center Extraction:
        Retrieve (x_px, y_px) from the bounding box registry produced by grounding.py.
        These coordinates are in the ORIGINAL IMAGE raster space (S_img).

    Stage 2 — Letterbox Compensation (Aspect-Ratio Correction):
        OmniParser's YOLO model resizes screenshots to a fixed square before inference.
        This padding (letterboxing) must be reversed to get true image-space coordinates.
        NOTE: ultralytics auto-reverses letterboxing in postprocessing, so Stage 2
              may be a no-op here — but we apply it explicitly per the directive.

        Math (Section 4, Stage 2):
            k       = min(model_input_size / img_width, model_input_size / img_height)
            delta_x = (model_input_size - k * img_width)  / 2
            delta_y = (model_input_size - k * img_height) / 2
            x_raster = (x_padded - delta_x) / k
            y_raster = (y_padded - delta_y) / k

    Stage 3 — DPR Logical Viewport Alignment:
        Screenshots are captured at physical resolution. Playwright clicks in logical
        (CSS) pixel space. Dividing by DPR converts between the two.

        Math (Section 4, Stage 3):
            x_logical = x_raster / DPR
            y_logical = y_raster / DPR

        WHY: On a 2x DPR screen (4K, Retina), physical pixel (800, 600) maps to
             logical coordinate (400, 300). Clicking at the physical value misses by 2x.

USAGE:
    from core.actuator import Actuator
    actuator = Actuator(page=engine.page, dpr=settings.DISPLAY_DPR)
    await actuator.execute(action_command, registry)
"""

import asyncio
import time
from typing import Any

from playwright.async_api import Page

from config import settings
from core.grounding import BoundingBoxRegistry
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class ActuatorError(Exception):
    """
    WHAT: Raised when an action cannot be executed (unknown action type,
          element not in registry, Playwright execution failure).
    WHY:  Typed exception enables main.py to catch and retry action failures
          without conflating them with perception or planning failures.
    """
    pass


class CoordinateError(ActuatorError):
    """
    WHAT: Raised specifically when the 3-stage coordinate transformation fails.
    WHY:  Coordinate math failures are a distinct failure mode — often caused by
          mismatched DPR settings or invalid registry data — and deserve a
          specific exception type for targeted debugging.
    """
    pass


class TaskCompletedError(Exception):
    """
    WHAT: Raised when the VLM issues the 'done' action, signaling task completion.
    WHY:  Python PEP 479 makes raising StopIteration inside an async function
          illegal — it gets wrapped in a fatal RuntimeError and crashes the
          program instead of cleanly exiting the loop. Using a custom exception
          avoids this and allows main.py to catch it cleanly.

    Attributes:
        reason: The VLM's explanation of why the task is complete.
    """
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


# ---------------------------------------------------------------------------
# Actuator — Main class
# ---------------------------------------------------------------------------

class Actuator:
    """
    WHAT: Translates VLM action commands into physical Playwright interactions
          by running coordinates through the 3-stage transformation pipeline.

    WHY:  Encapsulates all the coordinate math and Playwright API calls in one
          place, keeping main.py's loop focused on orchestration rather than
          the mechanics of clicking, typing, or scrolling.

    Attributes:
        _page: The active Playwright Page object (from PerceptionEngine).
        _dpr:  Device Pixel Ratio for Stage 3 (logical viewport alignment).
    """

    def __init__(self, page: Page) -> None:
        """
        WHAT: Initializes the Actuator with the Playwright page and DPR.
        WHY:  The Actuator needs the Page object to issue Playwright commands
              and the DPR to perform Stage 3 coordinate transformation.

        Args:
            page: Active Playwright Page from PerceptionEngine.page.
        """
        self._page: Page = page
        self._dpr: float = settings.DISPLAY_DPR

    # ------------------------------------------------------------------
    # 3-Stage Coordinate Pipeline
    # ------------------------------------------------------------------

    def _stage1_extract_center(
        self,
        registry: BoundingBoxRegistry,
        element_id: int,
    ) -> tuple[float, float, dict]:
        """
        WHAT: Stage 1 — Retrieves the bounding box center for element_id.

        WHY: The VLM returns an abstract element_id (not coordinates). Stage 1
             looks up the concrete pixel-space center from the grounding registry.

        Args:
            registry:   The bounding box registry from GroundingEngine.run().
            element_id: The numeric label selected by the VLM.

        Returns:
            tuple[float, float, dict]:
                - x_center: X coordinate in original image pixel space.
                - y_center: Y coordinate in original image pixel space.
                - entry:    The full registry entry (contains metadata for Stage 2).

        Raises:
            CoordinateError: If element_id is not found in the registry.
        """
        if element_id not in registry:
            raise CoordinateError(
                f"Element ID {element_id} not found in registry. "
                f"Available IDs: {list(registry.keys())}"
            )
        entry = registry[element_id]
        x_center = entry["x_center"]
        y_center = entry["y_center"]
        log.debug(
            f"[Stage 1] element_id={element_id} → "
            f"center=({x_center:.1f}, {y_center:.1f}) px (original image space)"
        )
        return x_center, y_center, entry

    def _stage2_letterbox_compensation(
        self,
        x_padded: float,
        y_padded: float,
        img_width: float,
        img_height: float,
        model_input_size: float,
    ) -> tuple[float, float]:
        """
        WHAT: Stage 2 — Reverses the letterbox padding applied by YOLO.

        WHY: When YOLO resizes a screenshot to its fixed square input size
             (e.g., 640×640), it adds gray padding to the shorter dimension.
             If coordinates come from the padded space, they must be
             un-letterboxed to get true original image coordinates.

        DERIVATION (from Section 4, Stage 2 of master directive):
            k       = min(model_input_size / img_width, model_input_size / img_height)
            delta_x = (model_input_size - k × img_width)  / 2
            delta_y = (model_input_size - k × img_height) / 2
            x_raster = (x_padded - delta_x) / k
            y_raster = (y_padded - delta_y) / k

            WHERE:
                k        = uniform scale factor applied by YOLO to fit image into square
                delta_x  = horizontal padding in pixels on each side of the padded image
                delta_y  = vertical padding in pixels on each side of the padded image

        NOTE:
            ultralytics YOLO postprocessing already un-letterboxes detection results
            before returning them. So the coordinates in our registry are already in
            original image space and this stage is effectively a no-op (k=1, delta=0).
            We apply it explicitly to remain faithful to the coordinate specification
            and to handle raw padded coordinates if OmniParser ever bypasses the
            standard ultralytics postprocessor.

        Args:
            x_padded:         X coordinate in padded model-input pixel space.
            y_padded:         Y coordinate in padded model-input pixel space.
            img_width:        Width of the ORIGINAL image before YOLO resize.
            img_height:       Height of the ORIGINAL image before YOLO resize.
            model_input_size: The YOLO model's fixed square input size (default 640).

        Returns:
            tuple[float, float]: (x_raster, y_raster) in original image raster space.
        """
        k: float = min(model_input_size / img_width, model_input_size / img_height)
        delta_x: float = (model_input_size - k * img_width) / 2.0
        delta_y: float = (model_input_size - k * img_height) / 2.0

        x_raster: float = (x_padded - delta_x) / k
        y_raster: float = (y_padded - delta_y) / k

        log.debug(
            f"[Stage 2] Letterbox compensation: k={k:.4f}, "
            f"Δx={delta_x:.1f}, Δy={delta_y:.1f} | "
            f"({x_padded:.1f}, {y_padded:.1f}) → ({x_raster:.1f}, {y_raster:.1f})"
        )
        return x_raster, y_raster

    def _stage3_dpr_logical_alignment(
        self,
        x_raster: float,
        y_raster: float,
    ) -> tuple[float, float]:
        """
        WHAT: Stage 3 — Converts physical raster coordinates to logical viewport coordinates.

        WHY: Playwright operates in logical (CSS) pixel space. Screenshots are captured
             at physical resolution (raster space). On HiDPI displays where DPR > 1.0,
             these two spaces differ by a factor of DPR.

        DERIVATION (from Section 4, Stage 3 of master directive):
            x_logical = x_raster / DPR
            y_logical = y_raster / DPR

            EXAMPLE (2x DPR Retina display):
                Physical screenshot pixel: (800, 600)
                DPR: 2.0
                Playwright logical click target: (400, 300)
                → Clicking at (800, 600) would miss by exactly 2x ← THIS IS THE BUG WE PREVENT

        Args:
            x_raster: X coordinate in physical image raster space.
            y_raster: Y coordinate in physical image raster space.

        Returns:
            tuple[float, float]: (x_logical, y_logical) in Playwright's CSS pixel space.
        """
        x_logical: float = x_raster / self._dpr
        y_logical: float = y_raster / self._dpr

        log.debug(
            f"[Stage 3] DPR={self._dpr} | "
            f"({x_raster:.1f}, {y_raster:.1f}) raster → "
            f"({x_logical:.1f}, {y_logical:.1f}) logical"
        )
        return x_logical, y_logical

    def transform_coordinates(
        self,
        registry: BoundingBoxRegistry,
        element_id: int,
    ) -> tuple[float, float]:
        """
        WHAT: Runs the full 3-stage coordinate transformation pipeline for element_id.

        WHY: Provides a single entry-point for the complete Stage 1 → 2 → 3
             transformation chain. This is what the execute() method calls internally.

        CRITICAL NOTE — Stage 2 bypass:
            ultralytics YOLO postprocessing already reverses letterbox padding
            before returning detection results. The coordinates in our registry
            are therefore already in original image pixel space (NOT padded space).
            Applying Stage 2 on top of that DOUBLES the transform — e.g., for a
            1280×720 viewport, k=0.5 would scale x from 639 to 1278, sending clicks
            to the extreme right edge of the screen.

            Stage 2 is retained in the codebase for documentation and for potential
            future use with raw OmniParser output, but is bypassed in the normal flow.

        Args:
            registry:   The grounding bounding box registry.
            element_id: The VLM-selected numeric element label.

        Returns:
            tuple[float, float]: Final (x, y) in Playwright logical CSS pixel space.

        Raises:
            CoordinateError: If any stage fails.
        """
        # Stage 1: Extract from registry (coordinates in original image pixel space)
        x_center, y_center, entry = self._stage1_extract_center(registry, element_id)

        # Stage 2: SKIPPED — ultralytics already un-letterboxes detection results.
        # The coordinates from grounding.py are already in original image space.
        # Applying letterbox compensation again would double the transform and
        # cause severe click drift (x=639 → 1278, y=207 → 134 on a 1280×720 viewport).
        x_raster = x_center
        y_raster = y_center

        log.debug(
            f"[Stage 2] BYPASSED — coords already in image space: "
            f"({x_raster:.1f}, {y_raster:.1f}) px"
        )

        # Stage 3: DPR logical alignment
        x_logical, y_logical = self._stage3_dpr_logical_alignment(x_raster, y_raster)

        log.act(
            f"Coordinate pipeline: element_id={element_id} → "
            f"logical=({x_logical:.1f}, {y_logical:.1f}) CSS px"
        )
        return x_logical, y_logical

    # ------------------------------------------------------------------
    # Action Executor — dispatch all supported action types
    # ------------------------------------------------------------------

    async def execute(
        self,
        action_command: dict[str, Any],
        registry: BoundingBoxRegistry,
    ) -> None:
        """
        WHAT: Dispatches a VLM action command to the appropriate Playwright handler.

        WHY: This is the final step of the Act phase. It reads the 'action' key
             from the command dict and routes to the correct Playwright method,
             after running coordinates through the 3-stage pipeline for element-based
             actions (click, type).

        SUPPORTED ACTIONS (Section 3, Phase 3 of master directive):
            click    → mouse.click(x_logical, y_logical)
            type     → element focus + keyboard.type(value)
            scroll   → mouse.wheel(delta_x, delta_y)
            keypress → keyboard.press(key)
            wait     → asyncio.sleep(seconds)
            done     → raises TaskCompletedError (signals loop exit)

        Args:
            action_command: The parsed JSON dict from Gemini's response.
                            Must contain 'action' key. May contain 'element_id',
                            'value', 'direction', 'amount', 'key', 'seconds', 'reason'.
            registry:       The bounding box registry for coordinate resolution.

        Raises:
            ActuatorError: If 'action' key is missing or the action type is unknown.
            CoordinateError: If coordinate transformation fails for element actions.
            TaskCompletedError: If action is 'done' (signals the main loop to exit cleanly).
        """
        action = action_command.get("action")
        if not action:
            raise ActuatorError(
                f"Action command missing 'action' key: {action_command}"
            )

        log.act(f"Executing action: {action_command}")

        if action == "click":
            await self._execute_click(action_command, registry)

        elif action == "type":
            await self._execute_type(action_command, registry)

        elif action == "scroll":
            await self._execute_scroll(action_command)

        elif action == "keypress":
            await self._execute_keypress(action_command)

        elif action == "wait":
            await self._execute_wait(action_command)

        elif action == "done":
            reason = action_command.get("reason", "Goal achieved.")
            log.act(f"✅ Agent completed task. Reason: {reason}")
            raise TaskCompletedError(reason)

        else:
            raise ActuatorError(
                f"Unknown action type: '{action}'. "
                f"Supported: click, type, scroll, keypress, wait, done."
            )

    # ------------------------------------------------------------------
    # Private action handlers
    # ------------------------------------------------------------------

    async def _execute_click(
        self,
        cmd: dict[str, Any],
        registry: BoundingBoxRegistry,
    ) -> None:
        """
        WHAT: Executes a left mouse click on the specified element.
        WHY:  Click is the most common GUI interaction — activates buttons,
              selects dropdowns, follows links.

        Args:
            cmd:      The full action command dict (must have 'element_id').
            registry: The bounding box registry for coordinate resolution.

        Raises:
            ActuatorError: If 'element_id' is missing from the command.
        """
        element_id = cmd.get("element_id")
        if element_id is None:
            raise ActuatorError("'click' action requires 'element_id' field.")

        x, y = self.transform_coordinates(registry, int(element_id))
        await self._page.mouse.click(x, y)
        log.act(f"🖱️  Left-click at logical ({x:.1f}, {y:.1f}) — element_id={element_id}")

    async def _execute_type(
        self,
        cmd: dict[str, Any],
        registry: BoundingBoxRegistry,
    ) -> None:
        """
        WHAT: Clicks an input element to focus it, then types the specified text.
        WHY:  Typing without first clicking the target element often sends keystrokes
              to the wrong focused element — especially on multi-input forms.

        Args:
            cmd:      The action command dict (must have 'element_id' and 'value').
            registry: The bounding box registry for coordinate resolution.

        Raises:
            ActuatorError: If 'element_id' or 'value' is missing.
        """
        element_id = cmd.get("element_id")
        value = cmd.get("value")

        if element_id is None:
            raise ActuatorError("'type' action requires 'element_id' field.")
        if value is None:
            raise ActuatorError("'type' action requires 'value' field.")

        # Click to focus the element
        x, y = self.transform_coordinates(registry, int(element_id))
        await self._page.mouse.click(x, y)
        await asyncio.sleep(0.1)  # Brief pause to ensure focus registers

        # Safely clear existing text ONLY if it's an input field.
        # WHY: A global 'Control+a' is dangerous. If the VLM hallucinates and clicks
        # a non-input element, Control+a selects the entire page's text (turning the
        # screen blue) which ruins the visual state for the next screenshot.
        await self._page.evaluate("""([x, y]) => {
            const el = document.elementFromPoint(x, y);
            if (!el) return;
            if (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA') {
                el.value = '';
                // Dispatch input event so React/Vue registers the clear
                el.dispatchEvent(new Event('input', { bubbles: true }));
            } else if (el.isContentEditable) {
                el.innerText = '';
            }
        }""", [x, y])
        await asyncio.sleep(0.05)

        await self._page.keyboard.type(str(value), delay=50)  # 50ms/char mimics human typing
        log.act(f"⌨️  Typed '{value}' into element_id={element_id}")

    async def _execute_scroll(self, cmd: dict[str, Any]) -> None:
        """
        WHAT: Scrolls the page in the specified direction by the given amount.
        WHY:  Many web UIs require scrolling to reveal content before it can
              be interacted with — e.g., lazy-loaded lists, virtual scroll tables.

        Args:
            cmd: Must have 'direction' ('up'/'down'/'left'/'right') and 'amount' (int).
                 Amount is in "scroll units" (1 unit ≈ 100px scroll delta).

        Raises:
            ActuatorError: If 'direction' is missing or invalid.
        """
        direction = cmd.get("direction", "down")
        amount = int(cmd.get("amount", 3))
        scroll_px = amount * 100  # Convert scroll units to pixels

        delta_x = 0
        delta_y = 0

        if direction == "down":
            delta_y = scroll_px
        elif direction == "up":
            delta_y = -scroll_px
        elif direction == "right":
            delta_x = scroll_px
        elif direction == "left":
            delta_x = -scroll_px
        else:
            raise ActuatorError(
                f"Invalid scroll direction: '{direction}'. "
                f"Valid: 'up', 'down', 'left', 'right'."
            )

        await self._page.mouse.wheel(delta_x, delta_y)
        log.act(f"🖱️  Scrolled {direction} by {amount} units ({scroll_px}px)")

    async def _execute_keypress(self, cmd: dict[str, Any]) -> None:
        """
        WHAT: Presses a keyboard key (e.g., Enter, Tab, Escape, ArrowDown).
        WHY:  Some UI interactions require keyboard navigation — submitting forms
              with Enter, dismissing dialogs with Escape, or tabbing between fields.

        Args:
            cmd: Must have 'key' — a Playwright key string (e.g., 'Enter', 'Tab').

        Raises:
            ActuatorError: If 'key' is missing.

        Reference:
            Playwright key names: https://playwright.dev/python/docs/api/class-keyboard
        """
        key = cmd.get("key")
        if not key:
            raise ActuatorError("'keypress' action requires 'key' field.")

        # Playwright requires strict title-case for keys (e.g., 'Enter', 'Tab', 'Escape')
        # If the VLM returns lowercase 'enter', this fixes it to avoid a crash.
        formatted_key = str(key)
        if len(formatted_key) > 0 and formatted_key[0].islower():
            # Only title-case single word keys to avoid messing up specific camelCase keys
            if formatted_key.lower() == "enter":
                formatted_key = "Enter"
            elif formatted_key.lower() == "tab":
                formatted_key = "Tab"
            elif formatted_key.lower() == "escape":
                formatted_key = "Escape"
            else:
                # Capitalize first letter as fallback
                formatted_key = formatted_key[0].upper() + formatted_key[1:]

        await self._page.keyboard.press(formatted_key)
        log.act(f"⌨️  Key pressed: '{formatted_key}'")

    async def _execute_wait(self, cmd: dict[str, Any]) -> None:
        """
        WHAT: Pauses the agent loop for the specified number of seconds.
        WHY:  Some UIs have animations, loading states, or rate-limiting that
              require the agent to wait before the next actionable state is ready.
              This is a deliberate, VLM-decided wait (not a stuck-state retry wait).

        Args:
            cmd: Must have 'seconds' (int). Capped at 30s to prevent abuse.

        Raises:
            ActuatorError: If 'seconds' is missing or invalid.
        """
        seconds = cmd.get("seconds")
        if seconds is None:
            raise ActuatorError("'wait' action requires 'seconds' field.")

        wait_time = min(float(seconds), 30.0)  # Cap at 30 seconds for safety
        log.act(f"⏳ Waiting {wait_time}s (VLM-requested pause)...")
        await asyncio.sleep(wait_time)
