# xLeRobot Web UI - Implementation Progress

## Architecture
- **Frontend**: Next.js 16 (App Router) + shadcn/ui + Tailwind CSS v4
- **Backend**: FastAPI (Python) + WebSocket for real-time logs
- **State**: React Context wizard state (client-side) + local JSON file (`webui_config.json`) for backend

---

## Backend (Complete)

### Services
- [x] `services/config_manager.py` - JSON config persistence (load/save/reset)
- [x] `services/process_manager.py` - Subprocess lifecycle (start/stop/logs/status)
- [x] `services/port_scanner.py` - Serial port detection via pyserial
- [x] `services/camera_scanner.py` - OpenCV camera detection + preview capture
- [x] `services/calibration_service.py` - Calibration file checking + command builder
- [x] `services/hf_service.py` - HuggingFace CLI integration (login/repos/create)

### Models
- [x] `models/config.py` - Config, CameraConfig, SingleArmConfig, BimanualConfig
- [x] `models/system.py` - ProcessStatus, CalibrationStatus, HFLoginStatus, SystemStatus
- [x] `models/setup.py` - PortInfo, CameraInfo, CameraPreview
- [x] `models/recording.py` - RecordingRequest, HFRepoInfo, CreateRepoRequest
- [x] `models/teleoperation.py` - TeleoperationRequest/Response

### API Routes
- [x] `api/config.py` - GET/POST/DELETE /api/config
- [x] `api/setup.py` - GET ports, GET cameras, POST camera previews, POST wiggle gripper, GET camera MJPEG stream
- [x] `api/calibration.py` - GET status, GET missing, GET files, POST start/stop
- [x] `api/teleoperation.py` - POST start/stop, GET status
- [x] `api/recording.py` - POST start/stop, GET status, DELETE cache
- [x] `api/huggingface.py` - GET whoami, GET/POST repos
- [x] `api/system.py` - GET system status

### Infrastructure
- [x] `main.py` - FastAPI app with CORS, static files, routers
- [x] `websockets/logs.py` - WebSocket log streaming endpoint
- [x] Entry point added to `pyproject.toml` (`lerobot-webui`)
- [x] `.gitignore` updated for webui files

---

## Frontend - Wizard Setup UI (Complete)

### Architecture
The frontend is a single-page wizard with 6 sequential steps. One centered card per step with a sidebar for navigation. Mock data layer with easy swap to real API.

### Core Infrastructure
- [x] `lib/wizard-types.ts` - All TypeScript interfaces, constants, initial state
- [x] `lib/services.ts` - Service layer with `USE_MOCK` toggle
- [x] `lib/mock-data.ts` - Mock responses for ports, cameras, calibration files
- [x] `lib/api.ts` - Real API client (used when `USE_MOCK=false`)
- [x] `lib/utils.ts` - Utility functions (cn)
- [x] `hooks/use-websocket.ts` - WebSocket hook for log streaming

### Wizard Components
- [x] `components/wizard/wizard-provider.tsx` - React Context + reducer for all wizard state
- [x] `components/wizard/wizard-layout.tsx` - Main shell: sidebar + topbar + centered card
- [x] `components/wizard/wizard-sidebar.tsx` - Step list with checkmarks, click navigation
- [x] `components/wizard/wizard-topbar.tsx` - "Clear Values" + "Restart" buttons
- [x] `components/wizard/step-card.tsx` - Shared card wrapper with Continue button

### Step Components
- [x] `components/wizard/steps/robot-type-step.tsx` - Step 1: Single/Bimanual selection cards
- [x] `components/wizard/steps/ports-step.tsx` - Step 2: Port scanning + role assignment + gripper wiggle to identify arms
- [x] `components/wizard/steps/cameras-step.tsx` - Step 3: Browser-native camera detection + live video feeds + naming
- [x] `components/wizard/steps/calibration-step.tsx` - Step 4: Calibration file selection per arm
- [x] `components/wizard/steps/teleoperate-step.tsx` - Step 5: Config summary + start/stop + logs
- [x] `components/wizard/steps/record-step.tsx` - Step 6: Recording form + start/stop + logs

### Shared Components (Reused)
- [x] `components/common/log-viewer.tsx` - Real-time log display with auto-scroll
- [x] `components/common/process-status.tsx` - Process status badge
- [x] `components/ui/` - 16 shadcn/ui components

### Pages
- [x] `app/layout.tsx` - Minimal root layout (fonts + TooltipProvider)
- [x] `app/page.tsx` - Single page wizard host with WizardProvider

---

## Changelog

### 2026-02-22
- **Feature: "No gripper detected" warning after port scan** — When the user clicks "Scan Ports" and no USB devices are found, the UI now shows a yellow warning banner with "No gripper detected" and troubleshooting tips (check power, re-plug USB, scan again). Previously the UI showed the same generic prompt as before scanning. Also added inline error display when the wiggle gripper action fails.
  - Modified: `frontend/components/wizard/steps/ports-step.tsx`

### 2026-02-21
- **Feature: New Calibration panel in wizard** — When user selects "+ New Calibration" for a role, an expandable panel appears below with: (1) a text input for the calibration file name (used as robot/teleoperator ID in teleoperation and recording), (2) a dashed image placeholder for a reference photo of the robot calibration position (to be added later), (3) a "Start Calibration" button that calls `POST /api/calibration/start` with the correct device type, ID, robot type, and port from wizard state. Includes real-time log viewer via WebSocket during calibration and a stop button. Only one calibration can run at a time across all roles. Step completion now requires a non-empty name when "new" is selected.
  - Modified: `frontend/components/wizard/steps/calibration-step.tsx`, `frontend/lib/wizard-types.ts`, `frontend/components/wizard/wizard-provider.tsx`, `frontend/lib/services.ts`
- **Feature: Auto-calibration backend** — Added `services/auto_calibration.py` with `AutoCalibrationService` that programmatically drives servos to find physical limits by detecting encoder stall. Algorithm: reset calibration, set homing offset, enable torque, step servo in one direction until encoder stops changing (stall detection), then reverse. Starts with gripper motor support. Added REST (`POST /api/calibration/auto/start`, `POST /api/calibration/auto/cancel`) and WebSocket (`/api/calibration/auto/ws`) endpoints for real-time progress streaming. Saves results to calibration JSON file and writes to motor EEPROM.
  - Created: `backend/services/auto_calibration.py`
  - Modified: `backend/api/calibration.py`

### 2025-02-20
- **Feature: Gripper wiggle for port identification** — Added POST `/api/setup/wiggle` endpoint that connects to a Feetech motor bus and wiggles the gripper servo (motor ID 6) using raw position values (no calibration needed). Frontend shows a hand icon button next to each port dropdown. Port assignment now supports auto-swap (changing one port swaps with the role that had it).
  - Modified: `backend/api/setup.py`, `frontend/components/wizard/steps/ports-step.tsx`, `frontend/components/wizard/wizard-provider.tsx`, `frontend/lib/services.ts`
- **Feature: Browser-native camera detection with live video feeds** — Replaced backend OpenCV camera detection with browser `navigator.mediaDevices.enumerateDevices()`. Camera labels now match actual devices (fixes system_profiler/OpenCV index mismatch). Live video feeds use `getUserMedia` with `<video>` elements. Built-in/phone cameras (FaceTime, iPhone, iPad, MacBook, IR) are automatically filtered out by label.
  - Modified: `frontend/components/wizard/steps/cameras-step.tsx`, `frontend/lib/wizard-types.ts`, `frontend/components/wizard/wizard-provider.tsx`, `frontend/lib/mock-data.ts`
- **Feature: Calibration files listing endpoint** — Added GET `/api/calibration/files` endpoint and `list_calibration_files` service method. Switched `USE_MOCK` to `false` in `services.ts` for real API integration.
  - Modified: `backend/api/calibration.py`, `backend/services/calibration_service.py`, `frontend/lib/services.ts`
- **Feature: MJPEG camera streaming endpoint** — Added GET `/api/setup/cameras/stream/{index}` with shared OpenCV capture per camera (auto-start/stop based on client connections). Not currently used by frontend (replaced by browser getUserMedia) but available for non-browser clients.
  - Modified: `backend/api/setup.py`

### 2025-02-18
- **Major: Replaced multi-page UI with wizard-style setup flow** — Inspired by Shopify's onboarding. Single-page app with 6 steps (Robot Type → Ports → Cameras → Calibration → Teleoperate → Record). One centered card per step, sidebar navigation with checkmarks, Clear Values + Restart buttons. Mock data layer with `USE_MOCK` flag for easy swap to real API.
  - Deleted: old multi-page routes (setup, calibration, teleoperation, recording), dashboard, old sidebar, mode-toggle, old types.ts
  - Created: wizard-provider (React Context + reducer), wizard-layout, wizard-sidebar, wizard-topbar, step-card, 6 step components, wizard-types, services layer, mock-data
  - Key features: step invalidation (changing early step resets later ones), port deduplication, camera name deduplication, calibration files filtered by robot type, config summary in teleoperate step, real-time logs via WebSocket

### 2025-02-17
- **Feature: Built-in camera filtering (macOS)** — Camera detection now uses `system_profiler SPCameraDataType -json` to identify built-in cameras. A "Hide built-in cameras" toggle (default: on) was added to the Setup page camera configuration section.

---

## How to Run

### Prerequisites
```bash
# Install backend dependencies
pip install fastapi uvicorn websockets python-multipart

# Install frontend dependencies
cd src/lerobot/webui/frontend
npm install
```

### Development
```bash
# Terminal 1: Start backend (port 8000)
cd /path/to/lerobot
python -m lerobot.webui.backend.main

# Terminal 2: Start frontend (port 3000)
cd src/lerobot/webui/frontend
npm run dev

# Open: http://localhost:3000
```

---

## File Inventory

```
src/lerobot/webui/
├── __init__.py
├── PROGRESS.md
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── calibration.py
│   │   ├── config.py
│   │   ├── huggingface.py
│   │   ├── recording.py
│   │   ├── setup.py
│   │   ├── system.py
│   │   └── teleoperation.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── recording.py
│   │   ├── setup.py
│   │   ├── system.py
│   │   └── teleoperation.py
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auto_calibration.py
│   │   ├── calibration_service.py
│   │   ├── camera_scanner.py
│   │   ├── config_manager.py
│   │   ├── hf_service.py
│   │   ├── port_scanner.py
│   │   └── process_manager.py
│   └── websockets/
│       ├── __init__.py
│       └── logs.py
└── frontend/
    ├── package.json
    ├── next.config.ts
    ├── app/
    │   ├── layout.tsx
    │   ├── page.tsx              (Wizard host)
    │   └── globals.css
    ├── components/
    │   ├── ui/                   (16 shadcn components)
    │   ├── common/
    │   │   ├── log-viewer.tsx
    │   │   └── process-status.tsx
    │   └── wizard/
    │       ├── wizard-provider.tsx
    │       ├── wizard-layout.tsx
    │       ├── wizard-sidebar.tsx
    │       ├── wizard-topbar.tsx
    │       ├── step-card.tsx
    │       └── steps/
    │           ├── robot-type-step.tsx
    │           ├── ports-step.tsx
    │           ├── cameras-step.tsx
    │           ├── calibration-step.tsx
    │           ├── teleoperate-step.tsx
    │           └── record-step.tsx
    ├── hooks/
    │   └── use-websocket.ts
    └── lib/
        ├── api.ts
        ├── wizard-types.ts
        ├── services.ts
        ├── mock-data.ts
        └── utils.ts
```
