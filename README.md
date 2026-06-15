# Pixel-to-Action GUI Autonomous Agent
> Vision-Driven Autonomous GUI Automation via Cloud Brain and Local Grounding

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)
![API](https://img.shields.io/badge/API-Gemini%20Cloud-orange.svg)

---

## 🚀 Live Demo

**[View Live Project / Agent in Action](https://drive.google.com/drive/folders/1mhvFSUbXp2H4cjHuqhQMzLPWAmJIPFJ8?usp=sharing)**

---

## Tech Stack Section

| Technology | Purpose |
| :--- | :--- |
| **Playwright** | Browser Automation (Local Body) for executing precise GUI interactions |
| **Gemini Cloud AI** | Cloud Brain responsible for reasoning, planning, and determining next actions |
| **OmniParser / YOLOv8** | Computer Vision backbone for local screen grounding and UI element detection |
| **PyTorch & OpenCV** | Image processing, tensor manipulation, and fast local model inference |
| **ImageHash** | pHash generation for perceptual state memory to prevent infinite loops |
| **Rich** | Advanced terminal UI providing beautiful, tagged execution logs |

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Configuration Reference](#configuration-reference)
- [Guardrails & Safety](#guardrails--safety)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

The Pixel-to-Action GUI Agent is a robust, vision-driven autonomous system designed to execute complex tasks directly through a graphical user interface without relying on underlying DOM structures. It solves the fragility of traditional web scrapers by combining a high-level cognitive cloud brain (Gemini) with a local computer vision grounding backbone (OmniParser). This novel architecture allows the agent to visually perceive, plan, and interact with user interfaces purely through pixels, granting it human-like versatility across any web application or platform.

---

## System Architecture

The core of the agent operates on a continuous, four-stage cognitive loop: **Perceive → Ground → Plan → Act**. 

1. **Perceive**: The system captures the current visual state of the environment and calculates a perceptual hash (pHash) to detect meaningful screen updates.
2. **Ground**: Local vision models (OmniParser/YOLO) detect interactive elements and overlay bounding boxes to create an annotated spatial representation.
3. **Plan**: The annotated visual state is sent to the Gemini Cloud AI, which analyzes the UI and formulates a discrete JSON-formatted action step.
4. **Act**: The system maps the planned action coordinates back to the original screen space and executes the interaction via Playwright.

```text
  ┌─────────────────────────────────────────────────────────┐
  │                   ORCHESTRATOR LOOP                     │
  └──────────────────────────┬──────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────┐
  │ 1. PERCEIVE                                             │
  │    Take Screenshot ────► pHash Compare ──► Store State  │
  └──────────────────────────┬──────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────┐
  │ 2. GROUND                                               │
  │    OmniParser (YOLO) ──► Bounding Boxes ──► Annotate UI │
  └──────────────────────────┬──────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────┐
  │ 3. PLAN                                                 │
  │    Annotated Screen ───► Gemini API ──────► JSON Action │
  └──────────────────────────┬──────────────────────────────┘
                             │
  ┌──────────────────────────▼──────────────────────────────┐
  │ 4. ACT                                                  │
  │    JSON Action ────────► Playwright ──────► Execute     │
  └──────────────────────────┬──────────────────────────────┘
                             │
                             └───────── Loop ───────────────┘
```

---

## Project Structure

```text
├── config/
│   ├── __init__.py           # Package initialization marker
│   ├── prompts.py            # Centralized system prompts for the Gemini planner
│   └── settings.py           # Strongly-typed environment configuration loader
├── core/
│   ├── __init__.py           # Package initialization marker
│   ├── actuator.py           # Translates planned JSON coordinates to Playwright commands
│   ├── grounding.py          # OmniParser integration for bounding box UI detection
│   ├── perception.py         # Screenshot capture and DPR handling logic
│   └── planner.py            # Interfaces with Gemini API to evaluate state and decide next steps
├── models/
│   ├── .cache/               # Local cache directory for downloaded model artifacts
│   ├── icon_detect/          # Custom icon detection models and sub-networks
│   └── README.txt            # Instructions on placement for local model weights
├── utils/
│   ├── __init__.py           # Package initialization marker
│   ├── logger.py             # Tagged Rich-based terminal UI logging framework
│   └── memory.py             # pHash state comparison for infinite loop detection
├── .env.example              # Template containing all configurable environment variables
├── .gitignore                # Git ignore rules for virtual environments and logs
├── main.py                   # Primary entrypoint orchestrating the Perceive-Ground-Plan-Act loop
├── requirements.txt          # Defined pip package dependencies
└── README.md                 # Primary project documentation
```

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-org/pixel-to-action-gui-agent.git
   cd pixel-to-action-gui-agent
   ```

2. **Set up the virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   playwright install
   ```

4. **Configure the environment variables:**
   Copy the example environment file and fill in your values.
   ```bash
   cp .env.example .env
   ```

### Environment Configuration

| Variable Name | Description | Example Value |
| :--- | :--- | :--- |
| `GEMINI_API_KEY` | Gemini Cloud API key | `AIzaSy...` |
| `GEMINI_MODEL_NAME` | Model identifier | `gemini-3.1-flash-lite` |
| `GEMINI_TEMPERATURE` | Generation temperature | `0.1` |
| `GEMINI_MAX_OUTPUT_TOKENS` | Output token limit | `256` |
| `MAX_STEPS` | Max execution steps | `20` |
| `DELTA_THRESHOLD` | Image diff threshold | `3` |
| `MAX_RETRIES` | Max error retries | `3` |
| `DISPLAY_DPR` | Device Pixel Ratio | `1.0` |
| `OMNIPARSER_MODEL_PATH` | Path to OmniParser model | `./models/omniparser_v2.pt` |
| `OMNIPARSER_MODEL_INPUT_SIZE` | Model input size | `640` |
| `LOG_LEVEL` | Terminal log level | `DEBUG` |
| `LOG_TO_FILE` | Enable file logging | `true` |
| `LOG_DIR` | Log directory path | `./logs` |

---

## Usage

Run the main orchestrator script to start the agent:

```bash
python main.py
```

### Example Terminal Output

```text
[PERCEIVE] Capturing current viewport state...
[PERCEIVE] pHash calculated: 8f3a2b1c4e5d6f7a
[GROUND] Extracting interactive elements via OmniParser...
[GROUND] Found 14 bounding boxes (confidence > 0.85).
[PLAN] Requesting next action from Gemini Cloud Brain...
[PLAN] Decided action: {"action": "click", "target_id": 4, "reason": "Submit login form"}
[ACT] Executing click at logical coordinates (x: 450, y: 320)
[ACT] Interaction successful. Awaiting visual response...
```

---

## Configuration Reference

The following parameters are controlled centrally via `config/settings.py` and sourced from `.env`:

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `GEMINI_API_KEY` | `str` | `""` | Gemini API key required for planning inference |
| `GEMINI_MODEL_NAME` | `str` | `"gemini-2.5-flash-preview-05-20"` | Production-stable Gemini model identifier |
| `GEMINI_TEMPERATURE` | `float` | `0.1` | Low temperature ensures near-deterministic JSON outputs |
| `GEMINI_MAX_OUTPUT_TOKENS` | `int` | `256` | Prevents runaway billing on a single planner call |
| `MAX_STEPS` | `int` | `20` | Hard ceiling on agent loop iterations to prevent runaway execution |
| `DELTA_THRESHOLD` | `int` | `3` | Hamming distance threshold for pHash state comparison |
| `MAX_RETRIES` | `int` | `3` | Maximum retry attempts with exponential backoff for API calls |
| `DISPLAY_DPR` | `float` | `1.0` | Device Pixel Ratio for Playwright coordinate mapping |
| `OMNIPARSER_MODEL_PATH` | `str` | `"./models/omniparser_v2.pt"` | Absolute path to the OmniParser v2 YOLO weights file |
| `OMNIPARSER_MODEL_INPUT_SIZE` | `int` | `640` | Square resolution the YOLO model resizes input images to |
| `LOG_LEVEL` | `str` | `"DEBUG"` | Terminal and file logging verbosity (`DEBUG`, `INFO`, etc.) |
| `LOG_TO_FILE` | `bool` | `True` | Whether to write logs to a file alongside terminal output |
| `LOG_DIR` | `str` | `"./logs"` | Target directory where `agent_*.log` files are stored |

---

## Guardrails & Safety

This agent operates autonomously and implements strict production safety guardrails to ensure safe execution:

1. **pHash State Memory**: The system calculates a perceptual hash (pHash) on every step. By computing the Hamming distance between consecutive states, it can reliably differentiate between actual UI updates and minor visual noise (like blinking cursors). If the distance is below the `DELTA_THRESHOLD`, the agent recognizes it is stuck and will halt.
2. **MAX_STEPS Limit**: A hard ceiling on loop iterations. Once the step count hits the predefined `MAX_STEPS` threshold, the execution forcefully terminates to prevent infinite loops and runaway cloud computing costs.
3. **Prompt Injection Defense**: The agent strictly separates the visual state from user inputs, relying entirely on structured JSON parsing from the Gemini planner. System prompts enforce strict constraints against executing untrusted arbitrary code or shell commands not predefined in the actuator's capabilities.

---

## Contributing

We welcome pull requests. For major changes, please open an issue first to discuss what you would like to change. Ensure all tests and linting checks pass before submitting your PR.

---

## License

MIT License
