"use client";

import {
  createContext,
  useContext,
  useReducer,
  useCallback,
  type ReactNode,
} from "react";
import type {
  WizardState,
  RobotMode,
  PortInfo,
  CameraInfo,
  CameraSelection,
  RecordingConfig,
} from "@/lib/wizard-types";
import {
  INITIAL_STATE,
  INITIAL_RECORDING_CONFIG,
  SINGLE_PORT_ROLES,
  BIMANUAL_PORT_ROLES,
} from "@/lib/wizard-types";

// Actions
type Action =
  | { type: "GO_TO_STEP"; step: number }
  | { type: "SET_ROBOT_MODE"; mode: RobotMode }
  | { type: "SET_DETECTED_PORTS"; ports: PortInfo[] }
  | { type: "SET_PORT_ASSIGNMENT"; role: string; port: string }
  | { type: "SET_DETECTED_CAMERAS"; cameras: CameraInfo[] }
  | { type: "SET_CAMERA_SELECTIONS"; selections: CameraSelection[] }
  | { type: "TOGGLE_CAMERA"; deviceId: string; included: boolean }
  | { type: "SET_CAMERA_NAME"; deviceId: string; name: string }
  | { type: "SET_CALIBRATION_FILES"; key: string; files: string[] }
  | { type: "SET_CALIBRATION_SELECTION"; role: string; filename: string | null }
  | { type: "SET_NEW_CALIBRATION_NAME"; role: string; name: string }
  | { type: "SET_TELE_PROCESS_ID"; id: string | null }
  | { type: "SET_RECORDING_CONFIG"; config: Partial<RecordingConfig> }
  | { type: "SET_RECORD_PROCESS_ID"; id: string | null }
  | { type: "CLEAR_ALL_VALUES" }
  | { type: "RESTART" };

// Step completion checker
function computeCompletedSteps(state: WizardState): boolean[] {
  const completed = [false, false, false, false, false, false];

  // Step 0: Robot Type
  completed[0] = state.robotMode !== null;

  // Step 1: Ports - all required roles assigned
  if (state.robotMode) {
    const roles =
      state.robotMode === "single" ? SINGLE_PORT_ROLES : BIMANUAL_PORT_ROLES;
    completed[1] = roles.every(
      (role) => state.portAssignments[role] && state.portAssignments[role] !== ""
    );
  }

  // Step 2: Cameras - at least 1 camera selected and named
  const selectedCameras = state.cameraSelections.filter((c) => c.included);
  completed[2] = selectedCameras.length > 0 && selectedCameras.every((c) => c.name !== "");

  // Step 3: Calibration - all roles have a selection
  if (state.robotMode) {
    const calRoles =
      state.robotMode === "single"
        ? ["follower", "leader"]
        : ["left_follower", "right_follower", "left_leader", "right_leader"];
    completed[3] = calRoles.every((role) => {
      const sel = state.calibrationSelections[role];
      if (sel === undefined || sel === null) return false;
      if (sel === "new") return (state.newCalibrationNames[role] || "").trim() !== "";
      return true;
    });
  }

  // Steps 4-5: always completable (operational)
  completed[4] = true;
  completed[5] = true;

  return completed;
}

// Reset steps from a given index onwards
function resetStepsFrom(state: WizardState, fromStep: number): WizardState {
  let s = { ...state };

  if (fromStep <= 1) {
    s.detectedPorts = [];
    s.portAssignments = {};
  }
  if (fromStep <= 2) {
    s.detectedCameras = [];
    s.cameraSelections = [];
  }
  if (fromStep <= 3) {
    s.calibrationFiles = {};
    s.calibrationSelections = {};
    s.newCalibrationNames = {};
  }
  if (fromStep <= 4) {
    s.teleProcessId = null;
  }
  if (fromStep <= 5) {
    s.recordingConfig = { ...INITIAL_RECORDING_CONFIG };
    s.recordProcessId = null;
  }

  s.completedSteps = computeCompletedSteps(s);
  return s;
}

function reducer(state: WizardState, action: Action): WizardState {
  let next: WizardState;

  switch (action.type) {
    case "GO_TO_STEP":
      next = { ...state, currentStep: action.step };
      break;

    case "SET_ROBOT_MODE": {
      // Changing robot type resets everything after step 0
      next = resetStepsFrom(
        { ...state, robotMode: action.mode },
        1
      );
      break;
    }

    case "SET_DETECTED_PORTS":
      next = { ...state, detectedPorts: action.ports };
      break;

    case "SET_PORT_ASSIGNMENT": {
      const newAssignments = { ...state.portAssignments };
      // If this port is already assigned to another role, swap them
      const previousPort = newAssignments[action.role] || "";
      for (const [otherRole, otherPort] of Object.entries(newAssignments)) {
        if (otherRole !== action.role && otherPort === action.port) {
          newAssignments[otherRole] = previousPort;
          break;
        }
      }
      newAssignments[action.role] = action.port;
      next = { ...state, portAssignments: newAssignments };
      break;
    }

    case "SET_DETECTED_CAMERAS":
      next = {
        ...state,
        detectedCameras: action.cameras,
        cameraSelections: action.cameras.map((c) => ({
          deviceId: c.deviceId,
          label: c.label,
          name: "",
          included: false,
        })),
      };
      break;

    case "SET_CAMERA_SELECTIONS":
      next = { ...state, cameraSelections: action.selections };
      break;

    case "TOGGLE_CAMERA":
      next = {
        ...state,
        cameraSelections: state.cameraSelections.map((c) =>
          c.deviceId === action.deviceId
            ? { ...c, included: action.included }
            : c
        ),
      };
      break;

    case "SET_CAMERA_NAME":
      next = {
        ...state,
        cameraSelections: state.cameraSelections.map((c) =>
          c.deviceId === action.deviceId ? { ...c, name: action.name } : c
        ),
      };
      break;

    case "SET_CALIBRATION_FILES":
      next = {
        ...state,
        calibrationFiles: {
          ...state.calibrationFiles,
          [action.key]: action.files,
        },
      };
      break;

    case "SET_CALIBRATION_SELECTION":
      next = {
        ...state,
        calibrationSelections: {
          ...state.calibrationSelections,
          [action.role]: action.filename,
        },
      };
      break;

    case "SET_NEW_CALIBRATION_NAME":
      next = {
        ...state,
        newCalibrationNames: {
          ...state.newCalibrationNames,
          [action.role]: action.name,
        },
      };
      break;

    case "SET_TELE_PROCESS_ID":
      next = { ...state, teleProcessId: action.id };
      break;

    case "SET_RECORDING_CONFIG":
      next = {
        ...state,
        recordingConfig: { ...state.recordingConfig, ...action.config },
      };
      break;

    case "SET_RECORD_PROCESS_ID":
      next = { ...state, recordProcessId: action.id };
      break;

    case "CLEAR_ALL_VALUES":
      next = { ...INITIAL_STATE, currentStep: state.currentStep };
      break;

    case "RESTART":
      next = { ...INITIAL_STATE };
      break;

    default:
      return state;
  }

  next.completedSteps = computeCompletedSteps(next);
  return next;
}

// Context
interface WizardContextValue {
  state: WizardState;
  dispatch: React.Dispatch<Action>;
  goToStep: (step: number) => void;
  goNext: () => void;
  clearAllValues: () => void;
  restart: () => void;
  allPriorStepsComplete: (step: number) => boolean;
}

const WizardContext = createContext<WizardContextValue | null>(null);

export function WizardProvider({ children }: { children: ReactNode }) {
  const [state, dispatch] = useReducer(reducer, INITIAL_STATE);

  const goToStep = useCallback(
    (step: number) => dispatch({ type: "GO_TO_STEP", step }),
    []
  );

  const goNext = useCallback(
    () =>
      dispatch({
        type: "GO_TO_STEP",
        step: Math.min(state.currentStep + 1, 5),
      }),
    [state.currentStep]
  );

  const clearAllValues = useCallback(
    () => dispatch({ type: "CLEAR_ALL_VALUES" }),
    []
  );

  const restart = useCallback(() => dispatch({ type: "RESTART" }), []);

  const allPriorStepsComplete = useCallback(
    (step: number) => {
      for (let i = 0; i < step; i++) {
        if (!state.completedSteps[i]) return false;
      }
      return true;
    },
    [state.completedSteps]
  );

  return (
    <WizardContext.Provider
      value={{
        state,
        dispatch,
        goToStep,
        goNext,
        clearAllValues,
        restart,
        allPriorStepsComplete,
      }}
    >
      {children}
    </WizardContext.Provider>
  );
}

export function useWizard() {
  const ctx = useContext(WizardContext);
  if (!ctx) throw new Error("useWizard must be used within WizardProvider");
  return ctx;
}
