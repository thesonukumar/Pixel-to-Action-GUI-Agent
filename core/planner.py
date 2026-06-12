"""
core/planner.py
===============
Gemini 3.5 Flash API integration — the Cloud Brain of the cognitive loop.

WHY THIS FILE EXISTS:
    The Plan phase (Phase 3 of the cognitive loop) sends the annotated screenshot
    and the user's goal to Gemini 3.5 Flash, then parses its JSON action response.

    This module handles:
      1. Gemini API client lifecycle (lazy initialization, key from settings)
      2. Multimodal content construction (image bytes + text prompt → Gemini)
      3. JSON parsing with markdown fence stripping (Gemini sometimes wraps JSON)
      4. Exponential backoff retry on API errors or JSONDecodeError (3 attempts)
      5. JSON correction prompt on parse failure (Section 10 of master directive)

    The Gemini model used: gemini-3.5-flash (production-stable, GA as of May 2026).
    Temperature: 0.1 for near-deterministic JSON output.
    Max tokens: 256 — actions are short JSON blobs, no prose needed.

    PROMPT ISOLATION RULE (Section 3):
        Gemini is instructed via SYSTEM_META_PROMPT to use ONLY OmniParser's numeric
        element IDs. It must NOT generate raw coordinates or perform spatial reasoning.
        This prevents coordinate system conflicts between OmniParser and Gemini.

USAGE:
    from core.planner import Planner
    planner = Planner()
    action_command = await planner.plan(
        screenshot_bytes=annotated_bytes,
        user_goal="Log in to the website",
        step_number=1,
        max_steps=20,
    )
    # action_command: {"action": "click", "element_id": 5}
"""

import asyncio
import json
import re
import time
from typing import Any, Optional

from google import genai
from google.genai import types

from config import settings, prompts
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Exceptions
# ---------------------------------------------------------------------------

class PlannerError(Exception):
    """
    WHAT: Raised when the Planner fails to produce a valid action after all retries.
    WHY:  Typed exception so main.py can distinguish planner failures from
          perception or actuation failures. The main loop can skip the current step
          or raise MaxStepsExceededError depending on policy.

    Attributes:
        attempts: Number of retry attempts made before giving up.
    """
    def __init__(self, message: str, attempts: int = 0) -> None:
        super().__init__(message)
        self.attempts = attempts


# ---------------------------------------------------------------------------
# Planner — Main class
# ---------------------------------------------------------------------------

class Planner:
    """
    WHAT: Calls the Gemini 3.5 Flash API with the annotated screenshot and user goal,
          parses the JSON action response, and returns a validated action command dict.

    WHY:  Encapsulates the entire Gemini interaction — client initialization, content
          construction, retry logic, and JSON parsing — behind a single async method.
          This keeps main.py's loop clean and decoupled from the specific LLM API.

    Attributes:
        _client:     The google-genai Client instance (lazy-initialized on first plan()).
        _model_name: The Gemini model identifier from settings.py.
    """

    def __init__(self) -> None:
        """
        WHAT: Configures the Planner with model parameters from settings.py.
        WHY:  Lazy initialization of the Gemini client avoids network connections
              at import time — the client is created on the first plan() call.
        """
        self._client: Optional[genai.Client] = None
        self._model_name: str = settings.GEMINI_MODEL_NAME

    def _get_client(self) -> genai.Client:
        """
        WHAT: Lazily initializes and caches the Gemini API client.
        WHY:  Creating the client once and reusing it across loop iterations
              avoids repeated authentication overhead and connection setup.
              The API key is read from settings.py (which sources it from .env).

        Returns:
            genai.Client: The initialized Gemini API client.

        Raises:
            PlannerError: If GEMINI_API_KEY is not configured.
        """
        if self._client is not None:
            return self._client

        if not settings.GEMINI_API_KEY:
            raise PlannerError(
                "GEMINI_API_KEY is not set. "
                "Cannot initialize Gemini client. Check your .env file."
            )

        self._client = genai.Client(api_key=settings.GEMINI_API_KEY)
        log.plan(f"Gemini client initialized — model: {self._model_name}")
        return self._client

    @staticmethod
    def _strip_markdown_fences(raw_text: str) -> str:
        """
        WHAT: Strips markdown code fences from Gemini's response if present.
        WHY:  Despite being instructed to return raw JSON, Gemini occasionally
              wraps the response in ```json ... ``` fences (especially when
              using lower temperatures). This method handles both:
                - ```json\n{...}\n```
                - ```\n{...}\n```
                - Raw JSON with no fences (most common case — returns as-is)

        DERIVATION:
            Regex pattern: ```(?:json)?\s*([\s\S]*?)\s*```
            Group 1 captures everything between the fences.
            The (?:json)? makes the language specifier optional.

        Args:
            raw_text: The raw string from response.text.

        Returns:
            str: The cleaned JSON string with fences removed.
        """
        # Try to extract JSON from markdown fences
        fence_pattern = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)
        match = fence_pattern.search(raw_text)
        if match:
            return match.group(1).strip()

        # No fences found — return cleaned text directly
        return raw_text.strip()

    @staticmethod
    def _validate_action_command(command: dict[str, Any]) -> None:
        """
        WHAT: Validates that the parsed JSON contains the required 'action' key
              and that the action type is one of the supported values.
        WHY:  Prevents downstream actuation failures from malformed VLM output.
              Better to catch and retry here than to crash the actuator mid-execution.

        Args:
            command: The parsed JSON dict from Gemini.

        Raises:
            ValueError: If 'action' key is missing or the action value is not supported.
        """
        supported_actions = {"click", "type", "scroll", "keypress", "wait", "done"}

        if "action" not in command:
            raise ValueError(
                f"Parsed JSON is missing required 'action' key. Got: {command}"
            )

        if command["action"] not in supported_actions:
            raise ValueError(
                f"Invalid action value: '{command['action']}'. "
                f"Supported actions: {supported_actions}"
            )

    async def plan(
        self,
        screenshot_bytes: bytes,
        user_goal: str,
        step_number: int,
        max_steps: int,
        action_history: list[dict] | None = None,
    ) -> dict[str, Any]:
        """
        WHAT: Sends the annotated screenshot + user goal to Gemini and returns
              a validated JSON action command dict.

        WHY:  This is the heart of the Plan phase. The multimodal Gemini call
              combines the visual evidence (annotated screenshot) with the semantic
              goal (user's task) and returns a grounded, actionable command.

        RETRY STRATEGY (Section 7 — Error Handling):
            - Max 3 attempts (settings.MAX_RETRIES)
            - Exponential backoff: 2s → 4s → 8s between attempts
            - On JSONDecodeError: retry with an explicit correction prompt (Section 10)
            - On API error: retry with the original prompt after backoff

        GEMINI CONTENT STRUCTURE (Section 10):
            contents=[
                Part.from_bytes(screenshot_png, mime_type="image/png"),
                Part.from_text(user_goal_prompt)
            ]
            config=GenerateContentConfig(
                system_instruction=SYSTEM_META_PROMPT,
                temperature=0.1,       # Near-deterministic JSON
                max_output_tokens=256  # Actions are short — cap for cost control
            )

        Args:
            screenshot_bytes: Annotated PNG screenshot bytes from GroundingEngine.
            user_goal:        The human-specified automation objective string.
            step_number:      Current loop iteration (for prompt context injection).
            max_steps:        MAX_STEPS ceiling (for prompt context injection).

        Returns:
            dict[str, Any]: A validated action command dict, e.g.:
                {"action": "click", "element_id": 14}
                {"action": "type",  "element_id": 5, "value": "user@example.com"}
                {"action": "done",  "reason": "Login complete"}

        Raises:
            PlannerError: If all retry attempts fail to produce a valid action command.
        """
        client = self._get_client()

        user_prompt = prompts.format_user_goal_prompt(
            user_goal=user_goal,
            step_number=step_number,
            max_steps=max_steps,
            action_history=action_history,
        )

        last_raw_response: Optional[str] = None
        max_attempts: int = settings.MAX_RETRIES

        for attempt in range(1, max_attempts + 1):
            backoff_delay = 2.0 ** (attempt - 1)  # 1s, 2s, 4s (shifted for first attempt)

            try:
                log.plan(
                    f"Calling Gemini API (attempt {attempt}/{max_attempts}) — "
                    f"model: {self._model_name}"
                )

                # Build content parts — correction prompt on retry after JSON failure
                if attempt > 1 and last_raw_response is not None:
                    # Retry with explicit JSON correction prompt (Section 10)
                    correction_prompt = prompts.format_json_correction_prompt(
                        previous_response=last_raw_response
                    )
                    text_part = types.Part.from_text(text=correction_prompt)
                    log.plan(f"Using JSON correction prompt for retry attempt {attempt}.")
                else:
                    text_part = types.Part.from_text(text=user_prompt)

                # --- Gemini API call (Section 10 pattern) ---
                response = client.models.generate_content(
                    model=self._model_name,
                    contents=[
                        types.Part.from_bytes(
                            data=screenshot_bytes,
                            mime_type="image/png",
                        ),
                        text_part,
                    ],
                    config=types.GenerateContentConfig(
                        system_instruction=prompts.SYSTEM_META_PROMPT,
                        temperature=settings.GEMINI_TEMPERATURE,
                        max_output_tokens=settings.GEMINI_MAX_OUTPUT_TOKENS,
                    ),
                )

                raw_text: str = response.text
                if not raw_text:
                    raise ValueError("API returned an empty response (possibly blocked by safety filters or rate limits).")
                last_raw_response = raw_text
                log.debug(f"[PLAN] Raw Gemini response: {raw_text!r}")

                # --- Strip markdown fences if present ---
                clean_json = self._strip_markdown_fences(raw_text)

                # --- Parse JSON ---
                command: dict[str, Any] = json.loads(clean_json)

                # --- Validate required structure ---
                self._validate_action_command(command)

                log.plan(f"✓ Valid action received: {command}")
                return command

            except json.JSONDecodeError as exc:
                log.error(
                    f"JSONDecodeError on attempt {attempt}/{max_attempts}. "
                    f"Raw response: {last_raw_response!r}. "
                    f"Retrying in {backoff_delay:.0f}s with correction prompt.",
                    exc_info=False,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(backoff_delay)
                continue

            except ValueError as exc:
                # Validation failure (wrong action type, missing keys)
                log.error(
                    f"Action validation failed on attempt {attempt}/{max_attempts}: {exc}. "
                    f"Retrying in {backoff_delay:.0f}s.",
                    exc_info=False,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(backoff_delay)
                continue

            except Exception as exc:
                # API-level errors (network, auth, quota)
                exc_str = str(exc)

                # Parse server-recommended retry delay for 429 rate-limit errors.
                # WHY: The Gemini free tier has a 5 RPM / 20 RPD limit. When hit,
                # the server returns "Please retry in 52.4s" but our default backoff
                # is only 1s→2s→4s, so we burn through all retries instantly and fail.
                # Extracting and using the actual delay lets us survive rate limits.
                actual_delay = backoff_delay
                if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str:
                    import re as _re
                    delay_match = _re.search(r"retry\s+in\s+([\d.]+)s", exc_str, _re.IGNORECASE)
                    if delay_match:
                        server_delay = float(delay_match.group(1))
                        # Use server delay + 2s buffer, capped at 90s
                        actual_delay = min(server_delay + 2.0, 90.0)
                        log.warning(
                            f"Rate-limited (429). Server says retry in {server_delay:.1f}s. "
                            f"Waiting {actual_delay:.1f}s before attempt {attempt + 1}/{max_attempts}."
                        )
                    else:
                        actual_delay = 30.0  # Safe default for rate limits
                        log.warning(
                            f"Rate-limited (429) but no retry delay found. "
                            f"Waiting {actual_delay:.0f}s before retry."
                        )

                log.error(
                    f"Gemini API error on attempt {attempt}/{max_attempts}: {exc}. "
                    f"Retrying in {actual_delay:.0f}s.",
                    exc_info=True,
                )
                if attempt < max_attempts:
                    await asyncio.sleep(actual_delay)
                continue

        # All attempts exhausted
        raise PlannerError(
            f"Failed to obtain a valid action from Gemini after {max_attempts} attempts. "
            f"Last raw response: {last_raw_response!r}",
            attempts=max_attempts,
        )
