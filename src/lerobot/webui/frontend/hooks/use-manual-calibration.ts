"use client";

import { useCallback, useRef, useState } from "react";

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export interface MotorValues {
  pos: number;
  min: number;
  max: number;
}

export type CalibrationPhase =
  | "disconnected"
  | "connecting"
  | "connected"
  | "homing"
  | "homing_done"
  | "recording"
  | "saving"
  | "saved"
  | "error";

export interface ManualCalibrationState {
  phase: CalibrationPhase;
  motors: string[];
  positions: Record<string, MotorValues>;
  savedPath: string | null;
  error: string | null;
}

const INITIAL_STATE: ManualCalibrationState = {
  phase: "disconnected",
  motors: [],
  positions: {},
  savedPath: null,
  error: null,
};

export function useManualCalibration() {
  const [state, setState] = useState<ManualCalibrationState>(INITIAL_STATE);
  const wsRef = useRef<WebSocket | null>(null);

  const send = useCallback((data: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  const connect = useCallback(
    (
      port: string,
      deviceType: string,
      robotType: string,
      deviceId: string
    ) => {
      // Clean up any existing connection
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }

      setState({ ...INITIAL_STATE, phase: "connecting" });

      const ws = new WebSocket(
        `${WS_BASE}/api/calibration/manual/ws`
      );
      wsRef.current = ws;

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            action: "start",
            port,
            device_type: deviceType,
            robot_type: robotType,
            device_id: deviceId,
          })
        );
      };

      ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);

        switch (msg.type) {
          case "connected":
            setState((s) => ({
              ...s,
              phase: "connected",
              motors: msg.motors,
              error: null,
            }));
            break;

          case "homing_done":
            setState((s) => ({ ...s, phase: "homing_done", error: null }));
            break;

          case "recording_started":
            setState((s) => ({ ...s, phase: "recording", error: null }));
            break;

          case "positions":
            setState((s) => ({
              ...s,
              positions: msg.motors,
            }));
            break;

          case "recording_done":
            setState((s) => ({ ...s, phase: "saving" }));
            break;

          case "saved":
            setState((s) => ({
              ...s,
              phase: "saved",
              savedPath: msg.path,
            }));
            break;

          case "error":
            setState((s) => ({
              ...s,
              phase: "error",
              error: msg.message,
            }));
            break;
        }
      };

      ws.onclose = () => {
        wsRef.current = null;
      };

      ws.onerror = () => {
        setState((s) => ({
          ...s,
          phase: "error",
          error: "WebSocket connection failed. Is the backend running?",
        }));
      };
    },
    []
  );

  const setHoming = useCallback(() => {
    setState((s) => ({ ...s, phase: "homing" }));
    send({ action: "set_homing" });
  }, [send]);

  const startRecording = useCallback(() => {
    send({ action: "start_recording" });
  }, [send]);

  const stopAndSave = useCallback(() => {
    send({ action: "stop_recording" });
  }, [send]);

  const disconnect = useCallback(() => {
    send({ action: "disconnect" });
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setState(INITIAL_STATE);
  }, [send]);

  const reset = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    setState(INITIAL_STATE);
  }, []);

  return { state, connect, setHoming, startRecording, stopAndSave, disconnect, reset };
}
