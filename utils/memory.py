"""
utils/memory.py
===============
Perceptual hash (pHash) state memory module — UI change detection and self-healing.

WHY THIS FILE EXISTS:
    Guardrail 1 from Section 5 of the master directive requires the agent to detect
    "stuck" UI states — situations where an action was executed but the screen did
    not visibly change (e.g., a click didn't register, the UI has a loading delay,
    or the agent clicked the wrong element).

    DETECTION STRATEGY — pHash vs Raw Pixel Diff:
        Raw pixel difference counts every changed pixel identically. Animated loading
        spinners, blinking cursors, and CSS transitions cause constant pixel churn —
        making raw diff useless as a "did the UI change meaningfully" metric.

        Perceptual Hash (pHash) works differently:
          1. Resize the image to 32×32 → convert to grayscale → apply DCT
          2. Compute the average of the DCT values
          3. Encode each pixel as 1 (above average) or 0 (below average)
          4. The result is a 64-bit fingerprint of the image's "perceptual content"

        Two images with the same visual structure (even with minor rendering noise)
        will have a Hamming distance close to 0. A meaningful UI change (new page,
        dialog appeared, button state changed) will shift the distance > DELTA_THRESHOLD.

    SELF-HEALING RETRY:
        If the Hamming distance after an action is < DELTA_THRESHOLD (default: 5),
        the UI did NOT change → the agent retries the same action up to MAX_RETRIES.
        After MAX_RETRIES consecutive stuck states → raise UIStuckError.

USAGE:
    from utils.memory import MemoryEngine, UIStuckError
    memory = MemoryEngine()
    hash_before = memory.compute_hash(screenshot_before_bytes)
    hash_after  = memory.compute_hash(screenshot_after_bytes)
    changed = memory.did_state_change(hash_before, hash_after)
    if not changed:
        memory.register_stuck_attempt()
        if memory.is_stuck():
            raise UIStuckError("UI appears frozen after 3 retries.")
    else:
        memory.reset_stuck_counter()
"""

import io
from dataclasses import dataclass, field
from typing import Optional

import imagehash
from PIL import Image

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class UIStuckError(Exception):
    """
    WHAT: Raised when the UI has not changed after MAX_RETRIES consecutive
          retry attempts following an action execution.
    WHY:  Surfaces stuck-UI scenarios as a typed exception so main.py can
          log the final state, cease retrying, and fail gracefully.

    Attributes:
        retry_count: The number of failed retry attempts before raising.
    """
    def __init__(self, message: str, retry_count: int = 0) -> None:
        super().__init__(message)
        self.retry_count = retry_count


# ---------------------------------------------------------------------------
# PerceptualHash wrapper — thin typed alias
# ---------------------------------------------------------------------------

PerceptualHash = imagehash.ImageHash
"""
Type alias for imagehash.ImageHash.
WHY: Provides semantic clarity — callers know they're working with a pHash
     and not a raw string or integer.
"""


# ---------------------------------------------------------------------------
# MemorySnapshot — stores a single state snapshot
# ---------------------------------------------------------------------------

@dataclass
class MemorySnapshot:
    """
    WHAT: Represents the memory state at one point in the cognitive loop.
    WHY:  Bundles the pHash with the step number for logging and debugging.

    Attributes:
        phash:       The perceptual hash of the screenshot.
        step_number: The agent loop step this snapshot was taken at.
    """
    phash: PerceptualHash
    step_number: int


# ---------------------------------------------------------------------------
# MemoryEngine — Main class
# ---------------------------------------------------------------------------

class MemoryEngine:
    """
    WHAT: Manages perceptual hash computation, Hamming distance comparison,
          stuck-state detection, and retry counting for the GUI automation agent.
    WHY:  Centralizes all state-change detection logic in one place, keeping
          main.py's loop clean and focused on orchestration.

    Attributes:
        _delta_threshold:  Max Hamming distance for "no change" classification.
        _max_retries:      Max consecutive stuck states before UIStuckError.
        _stuck_count:      Current count of consecutive stuck states.
        _last_snapshot:    The most recently computed MemorySnapshot.
        _history:          List of all snapshots (for debugging/replay).
    """

    def __init__(self) -> None:
        """
        WHAT: Initializes MemoryEngine with thresholds from settings.py.
        WHY:  All thresholds must come from settings — no magic numbers here.
        """
        self._delta_threshold: int = settings.DELTA_THRESHOLD
        self._max_retries: int = settings.MAX_RETRIES
        self._stuck_count: int = 0
        self._last_snapshot: Optional[MemorySnapshot] = None
        self._history: list[MemorySnapshot] = []

    # ------------------------------------------------------------------
    # Core pHash API
    # ------------------------------------------------------------------

    def compute_hash(self, screenshot_bytes: bytes) -> PerceptualHash:
        """
        WHAT: Computes the perceptual hash (pHash) of a screenshot.

        WHY: pHash encodes the "visual fingerprint" of the screen state in a
             64-bit integer. Two screenshots with the same layout structure
             will have a Hamming distance near 0, even if rendered at different
             times with minor pixel-level noise.

        DERIVATION:
            1. Decode raw PNG bytes → PIL Image
            2. Convert to grayscale (color changes don't indicate layout change)
            3. imagehash.phash() internally:
               a. Resizes image to hash_size² (default 32×32 after DCT)
               b. Applies 2D Discrete Cosine Transform (DCT)
               c. Retains only the top-left 8×8 low-frequency coefficients
               d. Encodes each coefficient as 1 (above mean) or 0 (below mean)
               e. Returns a 64-bit ImageHash object
            → The result is immune to minor animations, shadows, and cursor blink.

        Args:
            screenshot_bytes: Raw PNG screenshot bytes.

        Returns:
            PerceptualHash: The 64-bit perceptual hash of the image.

        Raises:
            ValueError: If screenshot_bytes cannot be decoded as an image.
        """
        try:
            image = Image.open(io.BytesIO(screenshot_bytes)).convert("L")  # Grayscale
            phash = imagehash.phash(image, hash_size=8)
            log.debug(f"[memory] Computed pHash: {phash}")
            return phash
        except Exception as exc:
            raise ValueError(f"Failed to compute pHash from screenshot: {exc}") from exc

    def hamming_distance(
        self,
        hash_a: PerceptualHash,
        hash_b: PerceptualHash,
    ) -> int:
        """
        WHAT: Computes the Hamming distance between two perceptual hashes.

        WHY: The Hamming distance counts the number of bit positions where the
             two hashes differ. A distance of 0 = identical images. A distance
             of 5 or less indicates visually similar states (no meaningful change).

        DERIVATION:
            hamming(A, B) = popcount(A XOR B)
            Where popcount = number of set bits in the XOR result.
            imagehash computes this via: len([b for b in (A.hash ^ B.hash).flatten() if b])

        Args:
            hash_a: pHash of the pre-action screenshot.
            hash_b: pHash of the post-action screenshot.

        Returns:
            int: Hamming distance (0 = identical, 64 = completely different).
        """
        distance = hash_a - hash_b  # imagehash overloads '-' as Hamming distance
        log.memory(
            f"Hamming distance: {distance} "
            f"(threshold: {self._delta_threshold}) — "
            f"{'STATE CHANGED ✓' if distance >= self._delta_threshold else 'NO CHANGE ✗'}"
        )
        return distance

    def did_state_change(
        self,
        hash_before: PerceptualHash,
        hash_after: PerceptualHash,
    ) -> bool:
        """
        WHAT: Determines if the UI state meaningfully changed between two screenshots.

        WHY: This is the core decision point for Guardrail 1. Returns True if the
             Hamming distance exceeds DELTA_THRESHOLD, indicating a real UI transition.
             Returns False if the screen looks essentially the same (stuck state).

        Args:
            hash_before: pHash computed BEFORE the action was executed.
            hash_after:  pHash computed AFTER the action + 500ms wait.

        Returns:
            bool: True if the UI changed meaningfully, False if stuck.
        """
        distance = self.hamming_distance(hash_before, hash_after)
        return distance >= self._delta_threshold

    # ------------------------------------------------------------------
    # Stuck-state counter management
    # ------------------------------------------------------------------

    def register_stuck_attempt(self) -> None:
        """
        WHAT: Increments the consecutive stuck-state counter.
        WHY:  Called whenever did_state_change() returns False. Tracking consecutive
              stuck states (not total stuck states) ensures the counter resets when
              the UI eventually responds — distinguishing "slow UI" from "broken UI".
        """
        self._stuck_count += 1
        log.memory(
            f"Stuck attempt {self._stuck_count}/{self._max_retries} registered. "
            f"{'→ Retrying action.' if not self.is_stuck() else '→ Raising UIStuckError.'}"
        )

    def reset_stuck_counter(self) -> None:
        """
        WHAT: Resets the stuck-state counter to zero after a successful UI change.
        WHY:  The counter must be reset whenever a state change is detected,
              so a single successful transition clears the stuck-state history.
              This allows the agent to handle "slow-loading" UIs without
              prematurely raising UIStuckError.
        """
        if self._stuck_count > 0:
            log.memory(f"UI responded — stuck counter reset (was {self._stuck_count}).")
        self._stuck_count = 0

    def is_stuck(self) -> bool:
        """
        WHAT: Returns True if the agent has been stuck for MAX_RETRIES consecutive steps.
        WHY:  This is the signal for main.py to raise UIStuckError rather than
              continuing to retry. Keeps the decision logic here, not in the loop.

        Returns:
            bool: True if stuck_count >= max_retries.
        """
        return self._stuck_count >= self._max_retries

    # ------------------------------------------------------------------
    # Snapshot management
    # ------------------------------------------------------------------

    def record_snapshot(
        self,
        screenshot_bytes: bytes,
        step_number: int,
    ) -> PerceptualHash:
        """
        WHAT: Computes a pHash, stores it as the latest snapshot, and appends
              it to the history list.

        WHY:  Provides a high-level API for main.py to record state at each step
              without managing raw hash objects directly. The history list enables
              post-run analysis and debugging of the agent's state transitions.

        Args:
            screenshot_bytes: Raw PNG screenshot bytes.
            step_number:      The current agent loop step (1-indexed).

        Returns:
            PerceptualHash: The computed hash (for immediate comparison if needed).
        """
        phash = self.compute_hash(screenshot_bytes)
        snapshot = MemorySnapshot(phash=phash, step_number=step_number)
        self._last_snapshot = snapshot
        self._history.append(snapshot)
        return phash

    @property
    def last_hash(self) -> Optional[PerceptualHash]:
        """
        WHAT: Returns the pHash from the most recent snapshot, or None.
        WHY:  Allows main.py to retrieve the pre-action hash without storing
              it in a local variable.

        Returns:
            Optional[PerceptualHash]: The last recorded hash, or None if no snapshot yet.
        """
        return self._last_snapshot.phash if self._last_snapshot else None

    @property
    def stuck_count(self) -> int:
        """
        WHAT: Returns the current consecutive stuck-state count.
        WHY:  Exposes internal state for logging in main.py without breaking
              encapsulation (caller cannot modify the counter directly).

        Returns:
            int: Current consecutive stuck state count.
        """
        return self._stuck_count

    @property
    def history(self) -> list[MemorySnapshot]:
        """
        WHAT: Returns the full list of recorded snapshots across all steps.
        WHY:  Enables post-run analysis of UI state transitions — useful for
              debugging why an agent got stuck at a particular step.

        Returns:
            list[MemorySnapshot]: All recorded snapshots in chronological order.
        """
        return self._history.copy()  # Return a copy to prevent external mutation
