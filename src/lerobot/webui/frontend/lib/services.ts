import type { PortInfo, CameraInfo, StartResponse, RecordingConfig, WizardState } from "./wizard-types";

const USE_MOCK = false;

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "";

/** Structured error with an optional traceback and developer hint, returned from the backend. */
export class DevError extends Error {
  traceback?: string;
  hint?: string;
  constructor(message: string, traceback?: string, hint?: string) {
    super(message);
    this.name = "DevError";
    this.traceback = traceback;
    this.hint = hint;
  }
}

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, options);
  } catch {
    throw new DevError(
      "Cannot connect to backend server",
      undefined,
      "The backend is not running. Start it with: python -m lerobot.webui.backend.main",
    );
  }
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    const detail = body?.detail;
    if (detail && typeof detail === "object") {
      throw new DevError(
        detail.message || `API error: ${res.status}`,
        detail.traceback,
        detail.hint,
      );
    }
    throw new Error(
      (typeof detail === "string" ? detail : null) ?? `API error: ${res.status}`
    );
  }
  return res.json();
}

export const services = {
  listPorts: async (): Promise<PortInfo[]> => {
    if (USE_MOCK) {
      const mock = await import("./mock-data");
      return mock.ports;
    }
    return fetchAPI<PortInfo[]>("/api/setup/ports");
  },

  listCameras: async (): Promise<CameraInfo[]> => {
    if (USE_MOCK) {
      const mock = await import("./mock-data");
      return mock.cameras;
    }
    return fetchAPI<CameraInfo[]>("/api/setup/cameras?exclude_builtin=true");
  },

  listCalibrationFiles: async (
    category: string,
    robotType: string
  ): Promise<string[]> => {
    if (USE_MOCK) {
      const mock = await import("./mock-data");
      return mock.calibrationFiles[`${category}/${robotType}`] || [];
    }
    return fetchAPI<string[]>(
      `/api/calibration/files?category=${encodeURIComponent(category)}&robot_type=${encodeURIComponent(robotType)}`
    );
  },

  startTeleoperation: async (
    displayData: boolean
  ): Promise<StartResponse> => {
    if (USE_MOCK) {
      const mock = await import("./mock-data");
      return mock.startResponse("teleoperation");
    }
    return fetchAPI<StartResponse>("/api/teleoperation/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ display_data: displayData }),
    });
  },

  stopProcess: async (processId: string): Promise<void> => {
    if (USE_MOCK) return;
    await fetchAPI(`/api/teleoperation/stop/${processId}`, {
      method: "POST",
    });
  },

  getProcessStatus: async (
    processId: string
  ): Promise<{
    process_id: string;
    process_type: string;
    state: "running" | "stopped" | "error";
    uptime_seconds: number | null;
    error_message: string | null;
  }> => {
    return fetchAPI(`/api/teleoperation/status/${processId}`);
  },

  startRecording: async (config: RecordingConfig): Promise<StartResponse> => {
    if (USE_MOCK) {
      const mock = await import("./mock-data");
      return mock.startResponse("recording");
    }
    return fetchAPI<StartResponse>("/api/recording/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repo_id: config.repoId,
        single_task: config.task,
        num_episodes: config.numEpisodes,
        episode_time_s: config.episodeTimeS,
        reset_time_s: config.resetTimeS,
        display_data: config.displayData,
      }),
    });
  },

  stopRecording: async (processId: string): Promise<void> => {
    if (USE_MOCK) return;
    await fetchAPI(`/api/recording/stop/${processId}`, { method: "POST" });
  },

  startCalibration: async (
    deviceType: string,
    deviceId: string,
    robotType: string,
    port: string
  ): Promise<StartResponse> => {
    if (USE_MOCK) {
      const mock = await import("./mock-data");
      return mock.startResponse("calibration");
    }
    return fetchAPI<StartResponse>("/api/calibration/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        device_type: deviceType,
        device_id: deviceId,
        robot_type: robotType,
        port,
      }),
    });
  },

  stopCalibration: async (processId: string): Promise<void> => {
    if (USE_MOCK) return;
    await fetchAPI(`/api/calibration/stop/${processId}`, { method: "POST" });
  },

  saveConfig: async (state: WizardState): Promise<void> => {
    if (USE_MOCK) return;
    const mode = state.robotMode === "bimanual" ? "bimanual" : "single";
    // Use the original detection index (position in cameraSelections) as the
    // OpenCV camera index, so excluded cameras don't shift the numbering.
    const cameras = state.cameraSelections
      .flatMap((c, i) =>
        c.included
          ? [{ index: i, name: c.name, width: 640, height: 480, fps: 30 }]
          : []
      );
    // Strip .json extension from calibration file names to get the ID
    const calId = (file: string | null | undefined) =>
      file && file !== "new" ? file.replace(/\.json$/, "") : null;

    const config =
      mode === "bimanual"
        ? {
            mode,
            bimanual: {
              left_follower_port: state.portAssignments.left_follower || null,
              left_leader_port: state.portAssignments.left_leader || null,
              right_follower_port: state.portAssignments.right_follower || null,
              right_leader_port: state.portAssignments.right_leader || null,
              left_follower_id: calId(state.calibrationSelections.left_follower),
              left_leader_id: calId(state.calibrationSelections.left_leader),
              right_follower_id: calId(state.calibrationSelections.right_follower),
              right_leader_id: calId(state.calibrationSelections.right_leader),
              cameras,
            },
          }
        : {
            mode,
            single_arm: {
              follower_port: state.portAssignments.follower || null,
              leader_port: state.portAssignments.leader || null,
              follower_id: calId(state.calibrationSelections.follower),
              leader_id: calId(state.calibrationSelections.leader),
              cameras,
            },
          };
    await fetchAPI("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(config),
    });
  },

  wiggleGripper: async (port: string): Promise<void> => {
    if (USE_MOCK) {
      await new Promise((r) => setTimeout(r, 2000));
      return;
    }
    await fetchAPI("/api/setup/wiggle", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ port }),
    });
  },

  clearCache: async (repoId: string): Promise<void> => {
    if (USE_MOCK) return;
    await fetchAPI(`/api/recording/cache?repo_id=${encodeURIComponent(repoId)}`, {
      method: "DELETE",
    });
  },

  openDataFolder: async (): Promise<void> => {
    if (USE_MOCK) return;
    await fetchAPI("/api/recording/open-folder", { method: "POST" });
  },
};
