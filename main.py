"""
main.py
=======
Central orchestrator for the Vision-Driven Pixel-to-Action GUI Automation Agent.

WHY THIS FILE EXISTS:
    This is the entry point of the agent. It implements the continuous
    "Perceive → Ground → Plan → Act" cognitive loop as a state machine that
    runs until one of three termination conditions is met:
      1. The VLM returns a "done" action (task successfully completed)
      2. step_count reaches MAX_STEPS (hard limit — prevents infinite billing)
      3. UIStuckError is raised (UI frozen after MAX_RETRIES retries)

    LOOP ARCHITECTURE (Section 3 of master directive):
        This is NOT a linear script. It is an event-driven loop where each
        iteration represents one full cognitive cycle of the agent:

        ┌─────────────────────────────────────────────┐
        │         Perceive → Ground → Plan → Act       │
        │                                             │
        │  1. Capture screenshot (PerceptionEngine)   │
        │  2. Record pHash BEFORE action (MemoryEngine)│
        │  3. Detect elements (GroundingEngine)        │
        │  4. Ask Gemini for action (Planner)          │
        │  5. Execute action (Actuator)                │
        │  6. Wait 500ms for UI to settle              │
        │  7. Capture screenshot AFTER action          │
        │  8. Compute pHash AFTER → compare Hamming   │
        │  9. If stuck → retry | else → next step      │
        └─────────────────────────────────────────────┘

    GUARDRAILS ACTIVE IN THIS LOOP:
        - Guardrail 1 (memory.py): pHash stuck-state detection + retry
        - Guardrail 2 (here):      MAX_STEPS hard ceiling
        - Guardrail 3 (prompts.py): Injection defense in system prompt

USAGE:
    python main.py --goal "Log in to GitHub with username admin and password secret" --url "https://github.com/login"

    Or via environment-configured defaults:
    python main.py --goal "Search for 'AI automation' on Google"

REQUIREMENTS:
    All dependencies in requirements.txt must be installed.
    .env must be configured (especially GEMINI_API_KEY).
    OmniParser model must be at OMNIPARSER_MODEL_PATH.
    Run `playwright install chromium` to install browser.
"""

import argparse
import asyncio
import sys
from typing import Optional

from config import settings
from core.actuator import Actuator, ActuatorError, CoordinateError, TaskCompletedError
from core.grounding import GroundingEngine, GroundingError
from core.perception import PerceptionEngine, PerceptionError
from core.planner import Planner, PlannerError
from utils.logger import get_logger, get_current_log_file
from utils.memory import MemoryEngine, UIStuckError

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class MaxStepsExceededError(Exception):
    """
    WHAT: Raised when the agent's step counter reaches MAX_STEPS.
    WHY:  Guardrail 2 (Section 5) requires the agent to raise this error
          and log the final state when the hard step ceiling is hit.
          Prevents infinite API billing loops on stuck or confused agents.

    Attributes:
        steps_taken: The number of steps executed before the ceiling was hit.
    """
    def __init__(self, message: str, steps_taken: int = 0) -> None:
        super().__init__(message)
        self.steps_taken = steps_taken


# ---------------------------------------------------------------------------
# Core Agent Loop
# ---------------------------------------------------------------------------

async def run_agent(
    user_goal: str,
    start_url: str,
    headless: bool = True,
    viewport_width: int = 1280,
    viewport_height: int = 720,
) -> None:
    """
    WHAT: The main async event loop — runs the full Perceive→Ground→Plan→Act
          cognitive cycle until task completion, step limit, or stuck-UI failure.

    WHY: Async is used because Playwright is an async library and the Gemini
         API calls are I/O bound. Running them in an async loop allows proper
         awaiting without blocking the event loop.

    Args:
        user_goal:       The automation objective in natural language.
        start_url:       The URL the browser should navigate to before looping.
        headless:        Whether to run the browser in headless mode.
        viewport_width:  Logical viewport width in CSS pixels.
        viewport_height: Logical viewport height in CSS pixels.

    Raises:
        MaxStepsExceededError: If MAX_STEPS is reached without task completion.
        UIStuckError:          If the UI is unresponsive after MAX_RETRIES retries.
        PerceptionError:       If the browser or screenshot mechanism fails.
    """
    # --- Validate configuration at startup (fail-fast) ---
    settings.validate()

    # --- Initialize engine instances ---
    grounding_engine = GroundingEngine()
    planner = Planner()
    memory = MemoryEngine()

    step_count: int = 0
    action_history: list[dict] = []  # Tracks past actions to prevent amnesia loops
    last_action_screenshot: Optional[bytes] = None  # Used for post-action pHash

    log.info(
        f"\n{'═'*60}\n"
        f"  🤖 Vision-Driven GUI Automation Agent — STARTING\n"
        f"  Goal: {user_goal}\n"
        f"  URL:  {start_url}\n"
        f"  Max Steps: {settings.MAX_STEPS} | DPR: {settings.DISPLAY_DPR}\n"
        f"  Log file: {get_current_log_file() or 'disabled'}\n"
        f"{'═'*60}"
    )

    async with PerceptionEngine(
        viewport_width=viewport_width,
        viewport_height=viewport_height,
        headless=headless,
    ) as engine:

        # --- Initialize Actuator with Playwright page ---
        actuator = Actuator(page=engine.page)

        # --- Navigate to starting URL ---
        await engine.navigate(start_url)

        # ──────────────────────────────────────────────────────────────
        # MAIN COGNITIVE LOOP
        # ──────────────────────────────────────────────────────────────
        while step_count < settings.MAX_STEPS:
            step_count += 1
            log.step_banner(step_count, settings.MAX_STEPS)

            # ── Guardrail 2: Check step ceiling before executing ──────
            if step_count > settings.MAX_STEPS:
                raise MaxStepsExceededError(
                    f"Agent exceeded MAX_STEPS={settings.MAX_STEPS}. "
                    f"Final URL: {await engine.get_current_url()}",
                    steps_taken=step_count,
                )

            # ─────────────────────────────────────────────────────────
            # PHASE 1: PERCEIVE — Capture raw screenshot
            # ─────────────────────────────────────────────────────────
            try:
                screenshot_bytes = await engine.capture()
                current_url = await engine.get_current_url()
                page_title = await engine.get_page_title()
                log.perceive(f"Page: '{page_title}' | URL: {current_url}")

            except PerceptionError as exc:
                log.error(f"Perception failed at step {step_count}: {exc}", exc_info=True)
                raise  # Perception failure is non-recoverable — propagate up

            # ─────────────────────────────────────────────────────────
            # GUARDRAIL 1 (Memory) — Record PRE-ACTION pHash
            # ─────────────────────────────────────────────────────────
            hash_before = memory.record_snapshot(screenshot_bytes, step_number=step_count)
            log.memory(f"Pre-action pHash recorded: {hash_before}")

            # ─────────────────────────────────────────────────────────
            # PHASE 2: GROUND — OmniParser element detection
            # ─────────────────────────────────────────────────────────
            try:
                annotated_bytes, registry = grounding_engine.run(screenshot_bytes)
                log.ground(f"Registry: {len(registry)} elements detected.")

            except GroundingError as exc:
                log.error(f"Grounding failed at step {step_count}: {exc}", exc_info=True)
                # Grounding failure is non-recoverable without model — propagate
                raise

            # ─────────────────────────────────────────────────────────
            # PHASE 3: PLAN — Gemini API call
            # ─────────────────────────────────────────────────────────
            try:
                action_command = await planner.plan(
                    screenshot_bytes=annotated_bytes,
                    user_goal=user_goal,
                    step_number=step_count,
                    max_steps=settings.MAX_STEPS,
                    action_history=action_history,
                )

            except PlannerError as exc:
                log.error(
                    f"Planner failed at step {step_count} after {exc.attempts} attempts: {exc}",
                    exc_info=True,
                )
                # Planner failure: log and skip to next step rather than crashing
                log.warning("Skipping this step and attempting to continue the loop.")
                memory.reset_stuck_counter()  # Don't carry stuck state across steps
                continue

            # ─────────────────────────────────────────────────────────
            # PHASE 4: ACT — Execute Playwright action
            # ─────────────────────────────────────────────────────────
            try:
                await actuator.execute(action_command, registry)

            except TaskCompletedError as done_signal:
                # "done" action received — task complete, exit the loop cleanly
                log.info(
                    f"\n{'═'*60}\n"
                    f"  ✅ TASK COMPLETE in {step_count} steps!\n"
                    f"  Reason: {done_signal.reason}\n"
                    f"{'═'*60}"
                )
                return  # Clean exit

            except Exception as exc:
                log.error(f"Actuation failed at step {step_count}: {exc}", exc_info=True)
                # Actuation failure: log and continue (may self-recover on next step)
                log.warning("Action failed — continuing loop, will re-perceive and re-plan.")
                memory.reset_stuck_counter()  # Don't carry stuck state across steps
                continue

            # ── Record successful action in history (Fix 2: Amnesia cure) ──
            action_history.append(action_command)

            # ─────────────────────────────────────────────────────────
            # GUARDRAIL 1 (Memory) — Post-action pHash comparison
            # ─────────────────────────────────────────────────────────
            # Fix 3: Skip pHash stuck-check ONLY for 'type' actions.
            # WHY: Typing changes so few pixels that pHash cannot detect the difference.
            # We do NOT skip it for 'keypress' because pressing 'Enter' often
            # triggers a full page navigation, which we MUST wait for and verify.
            action_type = action_command.get("action", "")
            if action_type == "type":
                log.memory(
                    f"Action '{action_type}' — bypassing pHash check "
                    f"(micro-interaction, visual delta too small for pHash)."
                )
                memory.reset_stuck_counter()
                await asyncio.sleep(0.3)  # Brief settle time for text rendering
                continue

            # Fix 4: Smart wait — use page load detection for click and keypress actions.
            # WHY: Clicks and keypresses (like Enter) frequently trigger heavy AJAX
            # or full page navigations. We must wait for the DOM to settle.
            if action_type in ("click", "keypress"):
                try:
                    await engine.page.wait_for_load_state(
                        "domcontentloaded", timeout=5000
                    )
                except Exception:
                    pass  # Action didn't trigger navigation; that's fine
                await asyncio.sleep(0.5)  # Additional settle time after load
            else:
                # scroll, wait, etc. — brief pause is sufficient
                await asyncio.sleep(0.5)

            try:
                post_action_bytes = await engine.capture()
            except PerceptionError:
                log.error("Post-action screenshot failed — skipping pHash check.")
                continue

            hash_after = memory.compute_hash(post_action_bytes)

            if not memory.did_state_change(hash_before, hash_after):
                # UI did NOT change → register stuck attempt
                memory.register_stuck_attempt()

                if memory.is_stuck():
                    raise UIStuckError(
                        f"UI appears frozen — no state change detected after "
                        f"{memory.stuck_count} consecutive retry attempts.\n"
                        f"Last action: {action_command}\n"
                        f"URL: {await engine.get_current_url()}",
                        retry_count=memory.stuck_count,
                    )
                else:
                    log.memory(
                        f"UI unchanged — will retry action. "
                        f"Attempt {memory.stuck_count}/{settings.MAX_RETRIES}"
                    )
                    # Decrement step counter so the retry doesn't "waste" a step
                    step_count -= 1
                    continue

            else:
                # UI changed successfully — reset stuck counter
                memory.reset_stuck_counter()
                log.memory("UI state updated successfully. ✓")

        # ──────────────────────────────────────────────────────────────
        # LOOP EXIT: MAX_STEPS reached without "done"
        # ──────────────────────────────────────────────────────────────
        final_url = await engine.get_current_url()
        raise MaxStepsExceededError(
            f"Agent reached MAX_STEPS={settings.MAX_STEPS} without completing the task.\n"
            f"Final URL: {final_url}",
            steps_taken=step_count,
        )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    WHAT: Parses command-line arguments for the agent.
    WHY:  Allows the agent to be invoked from the terminal with a goal and URL
          without editing any source code — the CLI is the correct interface
          for a production automation tool.

    Returns:
        argparse.Namespace: Parsed arguments object.
    """
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Vision-Driven Pixel-to-Action GUI Automation Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --goal "Log in to GitHub" --url "https://github.com/login"
  python main.py --goal "Search for AI news" --url "https://google.com" --visible
  python main.py --goal "Fill the contact form" --url "https://example.com/contact" --width 1920 --height 1080
        """,
    )
    parser.add_argument(
        "--goal",
        type=str,
        required=True,
        help="The automation goal in natural language (e.g., 'Log in to the website')",
    )
    parser.add_argument(
        "--url",
        type=str,
        required=True,
        help="The starting URL for the browser (e.g., 'https://example.com')",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        default=False,
        help="Run browser in visible (non-headless) mode. Useful for debugging.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Logical viewport width in CSS pixels (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Logical viewport height in CSS pixels (default: 720)",
    )
    return parser.parse_args()


def main() -> None:
    """
    WHAT: The synchronous entry point — parses CLI args and runs the async agent loop.
    WHY:  asyncio.run() creates a new event loop and runs the coroutine to completion,
          then cleans up the loop. This is the correct pattern for async entry points.
          Errors are caught here and reported cleanly to the terminal.
    """
    args = parse_args()

    try:
        asyncio.run(
            run_agent(
                user_goal=args.goal,
                start_url=args.url,
                headless=not args.visible,
                viewport_width=args.width,
                viewport_height=args.height,
            )
        )

    except MaxStepsExceededError as exc:
        log.error(
            f"\n{'═'*60}\n"
            f"  ⚠️  MAX STEPS EXCEEDED — Task NOT complete.\n"
            f"  Steps taken: {exc.steps_taken}/{settings.MAX_STEPS}\n"
            f"  {exc}\n"
            f"{'═'*60}"
        )
        sys.exit(1)

    except UIStuckError as exc:
        log.error(
            f"\n{'═'*60}\n"
            f"  🔒 UI STUCK — Agent cannot proceed.\n"
            f"  Retry attempts: {exc.retry_count}\n"
            f"  {exc}\n"
            f"{'═'*60}"
        )
        sys.exit(2)

    except PerceptionError as exc:
        log.error(
            f"\n{'═'*60}\n"
            f"  🖥️  BROWSER ERROR — Perception failure.\n"
            f"  {exc}\n"
            f"{'═'*60}",
            exc_info=True,
        )
        sys.exit(3)

    except KeyboardInterrupt:
        log.info("\n⚡ Agent interrupted by user (Ctrl+C). Shutting down gracefully.")
        sys.exit(0)

    except Exception as exc:
        log.error(f"Unexpected fatal error: {exc}", exc_info=True)
        sys.exit(99)


if __name__ == "__main__":
    main()
