"""
utils/logger.py
===============
Centralized Rich terminal + file logger for the Vision-Driven GUI Automation Agent.

WHY THIS FILE EXISTS:
    All phase transitions in the cognitive loop must be logged with specific
    prefixed tags: [PERCEIVE], [GROUND], [PLAN], [ACT], [MEMORY], [ERROR].
    Centralizing the logger ensures consistent formatting, color-coding, and
    log level control across every module — without each file managing its own
    logging configuration (Section 7, Code Quality Standards).

    DUAL OUTPUT STRATEGY:
        1. Rich Console  — Color-coded, human-friendly terminal output for live watching.
        2. Plain File    — Timestamped `.log` file for post-mortem debugging.
                           Strips ANSI color codes so the file is clean and grep-able.
                           One new file is created per agent run: agent_YYYY-MM-DD_HH-MM-SS.log

    Both outputs are active simultaneously. Toggle file logging via LOG_TO_FILE in .env.

USAGE:
    from utils.logger import get_logger
    log = get_logger(__name__)
    log.perceive("Captured screenshot at 1920x1080 px.")
    log.ground("OmniParser found 12 interactive elements.")
    log.plan("Gemini action → click element_id=5")
    log.act("Executing: left-click at (480, 320) logical px")
    log.memory("pHash Hamming distance=3 — UI state changed.")
    log.error("Gemini returned malformed JSON on attempt 1/3.")
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

from config import settings

# ---------------------------------------------------------------------------
# 1. Rich Console — shared singleton for terminal output
# ---------------------------------------------------------------------------

_AGENT_THEME = Theme(
    {
        "perceive": "bold cyan",
        "ground":   "bold yellow",
        "plan":     "bold magenta",
        "act":      "bold green",
        "memory":   "bold blue",
        "error":    "bold red",
        "info":     "white",
        "dim":      "dim white",
    }
)

_console = Console(theme=_AGENT_THEME, highlight=False)


# ---------------------------------------------------------------------------
# 2. ANSI strip helper — keeps log files clean and grep-able
# ---------------------------------------------------------------------------

_ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m|\[/?[a-z ]+\]")


def _strip_ansi(text: str) -> str:
    """
    WHAT: Removes ANSI color codes and Rich markup tags from a string.
    WHY:  Log files should be plain text — readable by any editor, grep,
          and log analysis tools. ANSI codes are meaningless outside a terminal.

    Args:
        text: The raw log message potentially containing ANSI/Rich markup.

    Returns:
        str: Clean plain-text string.
    """
    return _ANSI_ESCAPE.sub("", text)


# ---------------------------------------------------------------------------
# 3. Plain-text file formatter — strips color, adds timestamp + level
# ---------------------------------------------------------------------------

class _PlainFileFormatter(logging.Formatter):
    """
    WHAT: Custom formatter that strips ANSI/Rich markup and outputs clean
          plain-text log lines with ISO 8601 timestamps.
    WHY:  RichHandler produces beautifully colored terminal output, but those
          same escape sequences make log files unreadable. This formatter ensures
          the file output is always clean, structured, and tool-friendly.

    Format:
        2026-06-12 07:30:00.123 | INFO     | core.perception | [PERCEIVE] Screenshot captured.
    """

    def format(self, record: logging.LogRecord) -> str:
        """
        WHAT: Formats a log record into a plain-text line.
        WHY:  Overrides the default formatter to inject ISO timestamp and
              strip any Rich markup that leaked into the message.

        Args:
            record: The Python logging.LogRecord to format.

        Returns:
            str: A clean, plain-text log line.
        """
        timestamp = datetime.fromtimestamp(record.created).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        level     = record.levelname.ljust(8)
        name      = record.name
        message   = _strip_ansi(record.getMessage())

        line = f"{timestamp} | {level} | {name} | {message}"

        # Append exception traceback if present
        if record.exc_info:
            line += "\n" + self.formatException(record.exc_info)

        return line


# ---------------------------------------------------------------------------
# 4. Log level mapping
# ---------------------------------------------------------------------------

_LEVEL_MAP: dict[str, int] = {
    "DEBUG":   logging.DEBUG,
    "INFO":    logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR":   logging.ERROR,
}


# ---------------------------------------------------------------------------
# 5. Module-level log file path — set once per process on first get_logger()
# ---------------------------------------------------------------------------

_log_file_path: Optional[Path] = None


def _get_or_create_log_file() -> Optional[Path]:
    """
    WHAT: Returns the path of the current run's log file, creating it on first call.
    WHY:  All modules share ONE log file per agent run (not one per module).
          A module-level variable ensures the path is determined once and reused.
          The log directory is created if it doesn't exist.

    Returns:
        Optional[Path]: The log file path, or None if file logging is disabled.
    """
    global _log_file_path

    if not settings.LOG_TO_FILE:
        return None

    if _log_file_path is not None:
        return _log_file_path

    # Create logs directory if it doesn't exist
    log_dir = Path(settings.LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    # One timestamped file per agent run
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _log_file_path = log_dir / f"agent_{timestamp}.log"

    # Write a header to the new log file
    with open(_log_file_path, "w", encoding="utf-8") as f:
        f.write(f"{'='*70}\n")
        f.write(f"Vision-Driven GUI Automation Agent — Run Log\n")
        f.write(f"Started : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Log file: {_log_file_path}\n")
        f.write(f"{'='*70}\n\n")

    return _log_file_path


# ---------------------------------------------------------------------------
# 6. AgentLogger — wraps Python logging with phase-tagged Rich methods
# ---------------------------------------------------------------------------

class AgentLogger:
    """
    WHAT: A custom logger wrapper that exposes phase-specific logging methods
          (perceive, ground, plan, act, memory, error) on top of standard Python logging,
          with simultaneous Rich terminal output and plain-text file output.
    WHY:  The master directive (Section 7) requires every phase transition to be
          tagged with a recognizable prefix for observability and debugging.
          File logging adds production-grade post-mortem analysis capability.

    Attributes:
        _logger: The underlying Python logger instance.
        _name:   The module name this logger is bound to.
    """

    def __init__(self, name: str) -> None:
        """
        WHAT: Initializes the AgentLogger with both terminal and file handlers.
        WHY:  Called once per module — sets up a two-handler pipeline:
              Rich console (colored terminal) + FileHandler (plain text log file).

        Args:
            name: Usually __name__ from the calling module (e.g., 'core.perception').
        """
        self._name = name
        log_level = _LEVEL_MAP.get(settings.LOG_LEVEL.upper(), logging.DEBUG)

        # ── Handler 1: Rich terminal (colored, formatted for humans) ──────────
        rich_handler = RichHandler(
            console=_console,
            rich_tracebacks=True,
            show_path=True,
            markup=True,
        )
        rich_handler.setLevel(log_level)

        handlers: list[logging.Handler] = [rich_handler]

        # ── Handler 2: Plain file (clean text, grep-able, persistent) ─────────
        log_file = _get_or_create_log_file()
        if log_file is not None:
            file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
            file_handler.setLevel(log_level)
            file_handler.setFormatter(_PlainFileFormatter())
            handlers.append(file_handler)

        # Apply handlers to root logger (force=True overrides any existing setup)
        logging.basicConfig(
            level=log_level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=handlers,
            force=True,
        )

        self._logger = logging.getLogger(name)
        self._logger.setLevel(log_level)

        # Log the file path on first module init (so it's visible in both outputs)
        if log_file is not None and name == "__main__":
            self._logger.info(f"📄 Log file: {log_file}")

    # ------------------------------------------------------------------
    # Phase-tagged logging methods
    # ------------------------------------------------------------------

    def perceive(self, message: str, **kwargs) -> None:
        """
        WHAT: Logs a message for the Perceive phase (screenshot capture).
        WHY:  Tags output with [PERCEIVE] so operators can identify
              screenshot-related events in both terminal and log file.

        Args:
            message: Human-readable description of the perception event.
        """
        self._logger.info(
            f"[bold cyan]\\[PERCEIVE][/bold cyan] {message}", **kwargs
        )

    def ground(self, message: str, **kwargs) -> None:
        """
        WHAT: Logs a message for the Ground phase (OmniParser inference).
        WHY:  Tags output with [GROUND] to mark all bounding-box detection events.

        Args:
            message: Description of what was parsed from the screenshot.
        """
        self._logger.info(
            f"[bold yellow]\\[GROUND][/bold yellow] {message}", **kwargs
        )

    def plan(self, message: str, **kwargs) -> None:
        """
        WHAT: Logs a message for the Plan phase (Gemini API call and response).
        WHY:  Tags output with [PLAN] to trace every VLM decision in the loop.

        Args:
            message: Summary of the Gemini response or planning event.
        """
        self._logger.info(
            f"[bold magenta]\\[PLAN][/bold magenta] {message}", **kwargs
        )

    def act(self, message: str, **kwargs) -> None:
        """
        WHAT: Logs a message for the Act phase (Playwright execution).
        WHY:  Tags output with [ACT] to record every physical interaction
              the agent performs on the screen.

        Args:
            message: Description of the action being executed.
        """
        self._logger.info(
            f"[bold green]\\[ACT][/bold green] {message}", **kwargs
        )

    def memory(self, message: str, **kwargs) -> None:
        """
        WHAT: Logs a message for the Memory phase (pHash state comparison).
        WHY:  Tags output with [MEMORY] to surface UI state change detection events.

        Args:
            message: pHash comparison result or retry/stuck-state information.
        """
        self._logger.info(
            f"[bold blue]\\[MEMORY][/bold blue] {message}", **kwargs
        )

    def error(self, message: str, exc_info: bool = False, **kwargs) -> None:
        """
        WHAT: Logs an error message, with full traceback written to both
              terminal (Rich formatted) and log file (plain text).
        WHY:  File logging captures the full traceback permanently —
              critical for diagnosing failures in autonomous agent runs.

        Args:
            message:  Description of the failure.
            exc_info: If True, attaches the current exception traceback.
        """
        self._logger.error(
            f"[bold red]\\[ERROR][/bold red] {message}",
            exc_info=exc_info,
            **kwargs,
        )

    def info(self, message: str, **kwargs) -> None:
        """
        WHAT: Logs a general informational message not tied to a specific phase.
        WHY:  For startup/shutdown messages, step counter announcements,
              and other cross-phase observability events.

        Args:
            message: The informational message to log.
        """
        self._logger.info(message, **kwargs)

    def debug(self, message: str, **kwargs) -> None:
        """
        WHAT: Logs a debug-level message for deep diagnostic information.
        WHY:  File logging at DEBUG level captures raw API responses, pixel
              coordinates, and internal state — invaluable for debugging stuck agents.

        Args:
            message: The diagnostic message.
        """
        self._logger.debug(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """
        WHAT: Logs a warning-level message for non-fatal anomalies.
        WHY:  Warnings in the log file (e.g., nearing MAX_STEPS) give operators
              advance notice to intervene before a hard failure occurs.

        Args:
            message: The warning message.
        """
        self._logger.warning(message, **kwargs)

    def step_banner(self, step_number: int, max_steps: int) -> None:
        """
        WHAT: Prints a visually distinct separator for each agent step in terminal,
              and writes a plain-text equivalent to the log file.
        WHY:  Makes it trivial to find the start of each loop iteration in a
              long log stream — critical for debugging multi-step automation runs.

        Args:
            step_number: Current iteration (1-indexed).
            max_steps:   The MAX_STEPS ceiling.
        """
        # Rich terminal rule (looks great on screen)
        _console.rule(
            f"[bold white] STEP {step_number} / {max_steps} [/bold white]",
            style="dim white",
        )
        # Plain divider in log file (grep-able)
        self._logger.debug(
            f"{'─'*60} STEP {step_number}/{max_steps} {'─'*60}"
        )


# ---------------------------------------------------------------------------
# 7. Factory function — the public API for all modules
# ---------------------------------------------------------------------------

def get_logger(name: Optional[str] = None) -> AgentLogger:
    """
    WHAT: Returns an AgentLogger instance bound to the given module name.
    WHY:  Provides a single entry-point for logger acquisition — any module
          just calls `get_logger(__name__)` and gets a fully configured instance
          with both terminal and file output active.

    Args:
        name: Module name string (pass __name__ from the calling file).
              Defaults to 'agent' if not provided.

    Returns:
        AgentLogger: A fully configured phase-tagged logger with dual output.

    Example:
        from utils.logger import get_logger
        log = get_logger(__name__)
        log.perceive("Screenshot captured.")
    """
    return AgentLogger(name=name or "agent")


def get_current_log_file() -> Optional[Path]:
    """
    WHAT: Returns the path of the current run's log file.
    WHY:  Allows main.py to print the log file path in the startup banner
          and shutdown summary, so the user knows exactly where to find logs.

    Returns:
        Optional[Path]: The log file path, or None if file logging is disabled.
    """
    return _log_file_path
