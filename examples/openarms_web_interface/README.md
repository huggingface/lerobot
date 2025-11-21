# OpenArms Web Recording Interface

A web interface for recording OpenArms datasets.

## Installation

```bash
cd examples/openarms_web_interface
npm install
```

## Usage

**Start everything with one command:**

```bash
./launch.sh
```

This will:
- Start the FastAPI backend on port 8000
- Start the React frontend on port 5173
- Show live logs from both services

Then open your browser to: **http://localhost:5173**

**Stop with:** `Ctrl+C`

---

## Workflow

1. **Configure CAN interfaces** and **camera paths** in the dropdowns
2. Click **"Setup Robots"** to initialize (once at start)
3. Enter a **task description**
4. Click **"Start Recording"** to begin an episode
5. Click **"Stop Recording"** when done
6. Dataset is automatically encoded and uploaded to HuggingFace Hub as **private**
7. Repeat steps 3-6 for more episodes (no need to re-setup robots!)

---
