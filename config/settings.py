"""
config/settings.py
==================
Central configuration module for the Vision-Driven GUI Automation Agent.

WHY THIS FILE EXISTS:
    All runtime parameters — API keys, model names, thresholds, and file paths —
    are loaded from environment variables (via `.env`) and exposed as typed constants.
    This enforces the 'No Hardcoded Values' rule from Section 7 of the master directive.
    Logic files (core/, utils/) MUST import from here — never from `os.environ` directly.

USAGE:
    from config import settings
    print(settings.GEMINI_API_KEY)
    print(settings.MAX_STEPS)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Bootstrap: Load .env file from project root before reading any variables
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# 1. Gemini Cloud Brain
# ---------------------------------------------------------------------------

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
"""
Gemini API key sourced from .env.
NEVER hard-code this value in any other file.
"""

GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash-preview-05-20")
"""
Production-stable Gemini model identifier.
Defaults to gemini-2.5-flash-preview-05-20 as per Section 2 of master directive.
Swap model here without touching any logic file.
"""

GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.1"))
"""
Low temperature = near-deterministic JSON output from the planner.
0.1 is intentionally low to minimize hallucinations in action commands.
"""

GEMINI_MAX_OUTPUT_TOKENS: int = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "256"))
"""
Actions are short JSON blobs — cap tokens aggressively for cost control.
Prevents runaway billing on a single Gemini call (Section 10).
"""


# ---------------------------------------------------------------------------
# 2. Agent Execution Guardrails
# ---------------------------------------------------------------------------

MAX_STEPS: int = int(os.getenv("MAX_STEPS", "20"))
"""
Hard ceiling on agent loop iterations.
WHY: Prevents infinite API billing loops when the agent gets stuck (Guardrail 2).
If step_count >= MAX_STEPS, the orchestrator raises MaxStepsExceededError.
"""

DELTA_THRESHOLD: int = int(os.getenv("DELTA_THRESHOLD", "3"))
"""
Hamming distance threshold for pHash state comparison (Guardrail 1).
If hamming_distance(hash_before, hash_after) < DELTA_THRESHOLD → UI did not change.
pHash is used (not raw pixel diff) because it is stable against animated loaders
and blinking cursors.

WHY 3 (not 5): Typing text into a small input field only changes a few pixels in
a 1280×720 viewport, producing hamming distances of 3–4. A threshold of 5 causes
false "stuck" detections on every type action. A threshold of 3 still ignores
cursor blinks and rendering noise (distance 0–2) while correctly registering
keyboard input as a real state change.
"""

MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
"""
Maximum retry attempts for Gemini API calls and actuator self-healing.
Retry delays follow exponential backoff: 2s → 4s → 8s (Section 7).
"""


# ---------------------------------------------------------------------------
# 3. Display & Screenshot Parameters
# ---------------------------------------------------------------------------

DISPLAY_DPR: float = float(os.getenv("DISPLAY_DPR", "1.0"))
"""
Device Pixel Ratio of the host display.
CRITICAL: Playwright operates in logical layout pixels; screenshots are captured
at physical pixel resolution. All coordinate math depends on this value.
WHY NOT AUTO-DETECT: In headless mode, OS DPR does not propagate reliably
(Section 3, Phase 1 — Perceive).
"""


# ---------------------------------------------------------------------------
# 4. OmniParser Local Model (Grounding)
# ---------------------------------------------------------------------------

OMNIPARSER_MODEL_PATH: str = os.getenv(
    "OMNIPARSER_MODEL_PATH",
    str(_PROJECT_ROOT / "models" / "omniparser_v2.pt"),
)
"""
Absolute path to the OmniParser v2 YOLO weights file.
Must be downloaded separately and placed at this path before running.
"""

OMNIPARSER_MODEL_INPUT_SIZE: int = int(os.getenv("OMNIPARSER_MODEL_INPUT_SIZE", "640"))
"""
Internal square resolution the YOLO model resizes input images to.
WHY: OmniParser pads images to a square (letterboxing) before inference.
The actuator must reverse this padding to map bounding boxes back to
original image space (Letterbox Compensation — Stage 2 of coord pipeline).
Default of 640 matches standard YOLOv8 training resolution.
"""


# ---------------------------------------------------------------------------
# 5. Logging
# ---------------------------------------------------------------------------

LOG_LEVEL: str = os.getenv("LOG_LEVEL", "DEBUG")
"""
Logging verbosity. Passed to logger.py.
Options: DEBUG | INFO | WARNING | ERROR
"""

LOG_TO_FILE: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"
"""
If True, logs are written to a timestamped file inside LOG_DIR in addition
to the Rich terminal output. Both outputs are active simultaneously.
WHY: Terminal output disappears on close; file logs enable post-mortem
     debugging of multi-step autonomous agent runs.
"""

LOG_DIR: str = os.getenv("LOG_DIR", str(_PROJECT_ROOT / "logs"))
"""
Directory where log files are written. Created automatically if it doesn't exist.
Each agent run produces one file: agent_YYYY-MM-DD_HH-MM-SS.log
"""


# ---------------------------------------------------------------------------
# 6. Validation Guard (fail-fast on missing critical config)
# ---------------------------------------------------------------------------

def validate() -> None:
    """
    WHAT: Validates that all required environment variables are set.
    WHY: Surfaces misconfiguration early (at import time) rather than
         failing mid-loop during production execution.

    Raises:
        EnvironmentError: If GEMINI_API_KEY is missing.
    """
    if not GEMINI_API_KEY:
        raise EnvironmentError(
            "[settings] GEMINI_API_KEY is not set. "
            "Please add it to your .env file before running the agent."
        )
    if not Path(OMNIPARSER_MODEL_PATH).exists():
        # Warn, but do not crash — model may be downloaded later
        import warnings
        warnings.warn(
            f"[settings] OmniParser model not found at: {OMNIPARSER_MODEL_PATH}. "
            "The grounding module will fail unless the model is downloaded first.",
            RuntimeWarning,
            stacklevel=2,
        )
