"""
core/perception.py
==================
Playwright-based screen perception module for the Vision-Driven GUI Automation Agent.

WHY THIS FILE EXISTS:
    The Perceive phase (Phase 1 of the cognitive loop) is responsible for capturing
    a high-fidelity, correctly-scaled screenshot of the current browser state.

    CRITICAL ENGINEERING NOTE — DPR in Headless Mode:
        When Playwright runs in headless mode, the host OS Device Pixel Ratio (DPR)
        does NOT propagate automatically. If device_scale_factor is not explicitly set
        in new_context(), screenshots will be captured at logical resolution instead
        of physical resolution, causing all coordinate math to be wrong by a factor
        of DPR. We set it explicitly from settings.DISPLAY_DPR (Section 3, Phase 1).

    This module manages the full Playwright browser lifecycle (launch → context →
    page → screenshot → bytes) and exposes it via a clean async context manager.

USAGE:
    from core.perception import PerceptionEngine
    async with PerceptionEngine() as engine:
        await engine.navigate("https://example.com")
        screenshot_bytes = await engine.capture()
        width, height = engine.viewport_size
"""

import asyncio
from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
)

from config import settings
from utils.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class PerceptionError(Exception):
    """
    WHAT: Raised when the Perceive phase fails critically (browser crash,
          navigation error, screenshot failure).
    WHY:  Surfaces perception failures with a typed exception so the main
          loop can distinguish them from planning or actuation errors.
    """
    pass


# ---------------------------------------------------------------------------
# PerceptionEngine — Main class
# ---------------------------------------------------------------------------

class PerceptionEngine:
    """
    WHAT: Manages the Playwright browser lifecycle and exposes a screenshot API.
    WHY:  Encapsulates all browser state (playwright, browser, context, page) in
          one object so the main loop has a single, clean interface for perception.
          Using an async context manager guarantees proper resource cleanup even
          if an exception is raised mid-loop.

    Attributes:
        _dpr:        Device Pixel Ratio from settings — used for device_scale_factor.
        _playwright: The Playwright instance (created at __aenter__).
        _browser:    The launched Chromium browser instance.
        _context:    The browser context with explicit DPR configuration.
        _page:       The active browser page (tab).
        _viewport_w: Logical viewport width in CSS pixels.
        _viewport_h: Logical viewport height in CSS pixels.
    """

    def __init__(
        self,
        viewport_width: int = 1280,
        viewport_height: int = 720,
        headless: bool = True,
    ) -> None:
        """
        WHAT: Configures the PerceptionEngine before browser launch.
        WHY:  Separating configuration from launch allows the engine to be
              instantiated and inspected before entering the context manager.

        Args:
            viewport_width:  Logical width of the browser viewport in CSS pixels.
                             Default 1280 is a safe baseline for most web UIs.
            viewport_height: Logical height of the browser viewport in CSS pixels.
            headless:        If False, browser window is visible (useful for debugging).
                             Production runs use headless=True.
        """
        self._dpr: float = settings.DISPLAY_DPR
        self._viewport_w: int = viewport_width
        self._viewport_h: int = viewport_height
        self._headless: bool = headless

        # These are populated during __aenter__
        self._playwright: Optional[Playwright] = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None
        self._page: Optional[Page] = None

    # ------------------------------------------------------------------
    # Context Manager — enter / exit
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "PerceptionEngine":
        """
        WHAT: Launches the Playwright browser and creates a DPR-aware context.
        WHY:  __aenter__ is the logical place to acquire the browser resource,
              ensuring it's ready to use the moment the 'async with' block starts.

        CRITICAL DPR NOTE:
            device_scale_factor is explicitly set here from settings.DISPLAY_DPR.
            Do NOT remove this — headless Chromium defaults to 1.0 regardless of
            the host OS display, which causes all coordinate transformations to fail
            on HiDPI (Retina) displays where DPR > 1.0.

        Returns:
            PerceptionEngine: self (enables 'async with PerceptionEngine() as engine').

        Raises:
            PerceptionError: If the browser fails to launch.
        """
        try:
            self._playwright = await async_playwright().start()

            self._browser = await self._playwright.chromium.launch(
                headless=self._headless,
                args=[
                    "--disable-blink-features=AutomationControlled",  # Stealth
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                ],
            )

            # CRITICAL: Set device_scale_factor explicitly — DO NOT rely on OS DPR
            self._context = await self._browser.new_context(
                viewport={"width": self._viewport_w, "height": self._viewport_h},
                device_scale_factor=self._dpr,
                locale="en-US",
                timezone_id="America/New_York",
            )

            self._page = await self._context.new_page()

            log.perceive(
                f"Browser launched — viewport: {self._viewport_w}×{self._viewport_h} "
                f"logical px | DPR: {self._dpr} | headless: {self._headless}"
            )
            return self

        except Exception as exc:
            raise PerceptionError(
                f"Failed to launch Playwright browser: {exc}"
            ) from exc

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        WHAT: Gracefully shuts down the browser and Playwright instance.
        WHY:  Guarantees no zombie browser processes remain after the agent
              exits — even if an unhandled exception bubbles up from the loop.

        Args:
            exc_type: Exception type (None if no exception occurred).
            exc_val:  Exception value (None if no exception occurred).
            exc_tb:   Traceback (None if no exception occurred).
        """
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
            if self._playwright:
                await self._playwright.stop()
            log.perceive("Browser shut down cleanly.")
        except Exception as exc:
            log.error(f"Error during browser shutdown: {exc}", exc_info=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def navigate(self, url: str, wait_until: str = "domcontentloaded") -> None:
        """
        WHAT: Navigates the browser page to the specified URL.
        WHY:  Centralizes navigation logic so the main loop can trigger
              page loads without directly accessing the Playwright page object.

        Args:
            url:        The full URL to navigate to (e.g., 'https://example.com').
            wait_until: Playwright's wait strategy. 'domcontentloaded' waits for DOM
                        readiness — safe for all sites including heavy ones like Amazon.
                        'networkidle' (waits for zero network activity) often times out
                        on sites with continuous analytics/tracking requests.
                        Other options: 'load', 'commit'.

        Raises:
            PerceptionError: If navigation fails or the page is not initialized.
        """
        self._assert_ready()
        try:
            log.perceive(f"Navigating to: {url}")
            await self._page.goto(url, wait_until=wait_until, timeout=60_000)
            log.perceive(f"Navigation complete — page title: '{await self._page.title()}'")
        except Exception as exc:
            # If domcontentloaded also fails, try with 'commit' as a last resort
            if wait_until != "commit":
                try:
                    log.warning(
                        f"Navigation with '{wait_until}' failed. "
                        f"Retrying with 'commit' wait strategy..."
                    )
                    await self._page.goto(url, wait_until="commit", timeout=60_000)
                    # Give the page a moment to render after commit
                    await asyncio.sleep(2.0)
                    log.perceive(
                        f"Navigation complete (fallback) — "
                        f"page title: '{await self._page.title()}'"
                    )
                    return
                except Exception:
                    pass  # Fall through to the original error
            raise PerceptionError(f"Navigation failed for URL '{url}': {exc}") from exc

    async def capture(self) -> bytes:
        """
        WHAT: Takes a full-page screenshot and returns it as raw PNG bytes.
        WHY:  The screenshot is the raw perceptual input for OmniParser (Grounding)
              and Gemini (Planning). It must be in PNG format (lossless) to preserve
              fine text and UI element boundaries for accurate detection.

        PHYSICAL vs LOGICAL PIXELS:
            Playwright's screenshot() captures at PHYSICAL resolution when
            device_scale_factor > 1.0. For example, on a 2x DPR display, a
            1280×720 logical viewport produces a 2560×1440 physical screenshot.
            The actuator's Stage 3 (DPR division) reverses this for click coords.

        Returns:
            bytes: Raw PNG image data. Pass directly to OmniParser and Gemini.

        Raises:
            PerceptionError: If the screenshot fails.
        """
        self._assert_ready()
        try:
            screenshot_bytes: bytes = await self._page.screenshot(
                type="png",
                full_page=False,  # Capture viewport only (not full scrollable page)
            )
            # Physical dimensions for logging
            phys_w = int(self._viewport_w * self._dpr)
            phys_h = int(self._viewport_h * self._dpr)
            log.perceive(
                f"Screenshot captured — {phys_w}×{phys_h} physical px "
                f"({self._viewport_w}×{self._viewport_h} logical px @ DPR {self._dpr})"
            )
            return screenshot_bytes
        except Exception as exc:
            raise PerceptionError(f"Screenshot capture failed: {exc}") from exc

    async def get_page_title(self) -> str:
        """
        WHAT: Returns the current page title.
        WHY:  Useful for logging and for the planner to have page-level context.

        Returns:
            str: The current browser tab's document title.

        Raises:
            PerceptionError: If the page is not initialized.
        """
        self._assert_ready()
        return await self._page.title()

    async def get_current_url(self) -> str:
        """
        WHAT: Returns the current page URL.
        WHY:  Allows the planner and logger to record which URL the agent is on
              during each step — critical for multi-page navigation tasks.

        Returns:
            str: The current page URL string.

        Raises:
            PerceptionError: If the page is not initialized.
        """
        self._assert_ready()
        return self._page.url

    @property
    def viewport_size(self) -> tuple[int, int]:
        """
        WHAT: Returns the configured logical viewport dimensions.
        WHY:  The grounding module needs viewport dimensions to correctly compute
              the letterbox compensation offsets (Stage 2 of the coordinate pipeline).

        Returns:
            tuple[int, int]: (width, height) in logical CSS pixels.
        """
        return (self._viewport_w, self._viewport_h)

    @property
    def physical_size(self) -> tuple[int, int]:
        """
        WHAT: Returns the physical screenshot dimensions in raster pixels.
        WHY:  OmniParser processes the physical screenshot, so its bounding boxes
              are in physical pixel space. This property exposes the truth about
              what resolution OmniParser is actually working with.

        DERIVATION:
            physical_width  = viewport_width  × DPR
            physical_height = viewport_height × DPR

        Returns:
            tuple[int, int]: (width, height) in physical raster pixels.
        """
        return (
            int(self._viewport_w * self._dpr),
            int(self._viewport_h * self._dpr),
        )

    @property
    def page(self) -> Page:
        """
        WHAT: Exposes the raw Playwright Page object for the actuator.
        WHY:  The actuator (core/actuator.py) needs direct access to the Page
              to call page.mouse.click(), page.keyboard.type(), etc.
              Exposing it as a property keeps PerceptionEngine the single owner
              while allowing controlled access.

        Returns:
            Page: The active Playwright page instance.

        Raises:
            PerceptionError: If called before __aenter__.
        """
        self._assert_ready()
        return self._page

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _assert_ready(self) -> None:
        """
        WHAT: Asserts that the browser and page are initialized before any operation.
        WHY:  Prevents cryptic AttributeError messages if methods are called
              before entering the async context manager.

        Raises:
            PerceptionError: If the PerceptionEngine is not in an active context.
        """
        if self._page is None:
            raise PerceptionError(
                "PerceptionEngine is not active. "
                "Use 'async with PerceptionEngine() as engine:' before calling methods."
            )
