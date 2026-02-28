"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  Camera,
  ChevronDown,
  ChevronRight,
  CircleCheck,
  Loader2,
  Play,
  Square,
  XCircle,
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { LogViewer } from "@/components/common/log-viewer";
import { useWebSocket } from "@/hooks/use-websocket";
import { services } from "@/lib/services";
import { useWizard } from "../wizard-provider";
import { StepCard } from "../step-card";

type TeleState = "idle" | "starting" | "running" | "error" | "stopped";

// Regex to match motor position lines like "shoulder_pan.pos        |  12.45"
const MOTOR_LINE_RE = /^([\w.]+)\s+\|\s+([-+]?\d+\.\d+)\s*$/;
// Regex to match timing lines like "time: 16.67ms (60 Hz)"
const TIME_LINE_RE = /^time:\s+([\d.]+)ms\s+\((\d+)\s+Hz\)/;

const MOTOR_RANGE = 100; // Normalized motor positions are roughly in [-100, 100]

/** Stable motor state: only updates at a throttled rate to avoid glitching. */
function useMotorState(logs: string[], isRunning: boolean) {
  const [motors, setMotors] = useState<Record<string, number>>({});
  const [motorOrder, setMotorOrder] = useState<string[]>([]);
  const [frequency, setFrequency] = useState<number | null>(null);
  const lastUpdateRef = useRef(0);
  const prevLogLenRef = useRef(0);

  useEffect(() => {
    if (!isRunning) return;

    // Only parse new logs since last check
    const now = performance.now();
    if (now - lastUpdateRef.current < 80) return; // Throttle to ~12Hz
    lastUpdateRef.current = now;

    // Parse only the tail of logs (last 20 lines is enough for one cycle)
    const tail = logs.slice(-20);
    const parsed: Record<string, number> = {};
    let freq: number | null = null;

    for (let i = tail.length - 1; i >= 0; i--) {
      const line = tail[i];

      const timeMatch = line.match(TIME_LINE_RE);
      if (timeMatch && freq === null) {
        freq = parseInt(timeMatch[2], 10);
      }

      const motorMatch = line.match(MOTOR_LINE_RE);
      if (motorMatch && !(motorMatch[1] in parsed)) {
        parsed[motorMatch[1]] = parseFloat(motorMatch[2]);
      }

      if (Object.keys(parsed).length >= 12 && freq !== null) break;
    }

    if (Object.keys(parsed).length > 0) {
      setMotors(parsed);
      // Lock in motor order on first parse so entries don't jump around
      setMotorOrder((prev) =>
        prev.length > 0 ? prev : Object.keys(parsed)
      );
    }
    if (freq !== null) setFrequency(freq);
    prevLogLenRef.current = logs.length;
  }, [logs, isRunning]);

  return { motors, motorOrder, frequency };
}

/** Single motor row with a visual position bar and numeric value. */
function MotorRow({ name, value }: { name: string; value: number }) {
  // Map value from [-MOTOR_RANGE, MOTOR_RANGE] to [0%, 100%]
  const pct = Math.min(
    100,
    Math.max(0, ((value + MOTOR_RANGE) / (2 * MOTOR_RANGE)) * 100)
  );

  return (
    <div className="flex items-center gap-3 px-3 py-1.5">
      <span className="w-28 shrink-0 text-xs text-muted-foreground truncate">
        {name.replace(".pos", "")}
      </span>
      <div className="relative flex-1 h-2 rounded-full bg-muted/50 overflow-hidden">
        {/* Center line at 0 */}
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-muted-foreground/20" />
        {/* Fill from center to current value */}
        {value >= 0 ? (
          <div
            className="absolute top-0 bottom-0 left-1/2 rounded-r-full bg-emerald-500/80 transition-all duration-75"
            style={{ width: `${(pct - 50)}%` }}
          />
        ) : (
          <div
            className="absolute top-0 bottom-0 right-1/2 rounded-l-full bg-emerald-500/80 transition-all duration-75"
            style={{ width: `${(50 - pct)}%` }}
          />
        )}
        {/* Thumb indicator */}
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 h-3 w-3 rounded-full bg-emerald-400 border-2 border-background shadow-sm transition-all duration-75"
          style={{ left: `${pct}%` }}
        />
      </div>
      <span className="w-16 shrink-0 text-right text-xs font-mono tabular-nums text-foreground">
        {value.toFixed(2)}
      </span>
    </div>
  );
}

/** Live motor position display with slider bars. */
function MotorPanel({
  motors,
  motorOrder,
  frequency,
}: {
  motors: Record<string, number>;
  motorOrder: string[];
  frequency: number | null;
}) {
  if (motorOrder.length === 0) return null;

  return (
    <div className="rounded-lg border bg-card">
      <div className="px-3 py-2 border-b">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium text-muted-foreground">
            Motor Positions
          </span>
          {frequency !== null && (
            <span className="text-[10px] text-muted-foreground/60">
              {frequency} Hz
            </span>
          )}
        </div>
      </div>
      <div className="py-1 divide-y divide-border/50">
        {motorOrder.map((name) => (
          <MotorRow key={name} name={name} value={motors[name] ?? 0} />
        ))}
      </div>
    </div>
  );
}

/** Single camera feed using the browser getUserMedia API (same as cameras step). */
function CameraFeed({ deviceId }: { deviceId: string }) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function start() {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { deviceId: { exact: deviceId } },
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        console.error("Failed to open camera", deviceId, err);
      }
    }

    start();

    return () => {
      cancelled = true;
      streamRef.current?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, [deviceId]);

  return (
    <video
      ref={videoRef}
      autoPlay
      playsInline
      muted
      className="w-full rounded-lg"
    />
  );
}

/** Live camera feed panel showing the cameras the user selected in the cameras step. */
function CameraFeedPanel({
  cameras,
}: {
  cameras: { deviceId: string; name: string }[];
}) {
  if (cameras.length === 0) return null;

  return (
    <div className="rounded-lg border bg-card">
      <div className="px-3 py-2 border-b">
        <div className="flex items-center gap-2">
          <Camera className="h-3.5 w-3.5 text-muted-foreground" />
          <span className="text-xs font-medium text-muted-foreground">
            Camera Feeds
          </span>
        </div>
      </div>
      <div className="space-y-3 p-3">
        {cameras.map((cam) => (
          <div key={cam.deviceId} className="space-y-1.5">
            <span className="text-xs text-muted-foreground font-medium">
              {cam.name}
            </span>
            <CameraFeed deviceId={cam.deviceId} />
          </div>
        ))}
      </div>
    </div>
  );
}

export function TeleoperateStep() {
  const { state, dispatch, allPriorStepsComplete } = useWizard();
  const [teleState, setTeleState] = useState<TeleState>(
    state.teleProcessId ? "running" : "idle"
  );
  const [showLogs, setShowLogs] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [showCameras, setShowCameras] = useState(false);
  const priorComplete = allPriorStepsComplete(4);

  const selectedCameraFeeds = state.cameraSelections
    .filter((c) => c.included && c.name)
    .map((c) => ({ deviceId: c.deviceId, name: c.name }));

  const { logs, isConnected, clearLogs } = useWebSocket(state.teleProcessId);

  // Poll process status to detect crashes
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current) {
      clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  const startPolling = useCallback(
    (processId: string) => {
      stopPolling();
      pollRef.current = setInterval(async () => {
        try {
          const status = await services.getProcessStatus(processId);
          if (status.state === "error") {
            setTeleState("error");
            setErrorMsg(status.error_message || "Process exited with an error");
            setShowLogs(true);
            stopPolling();
          } else if (status.state === "stopped") {
            setTeleState("stopped");
            stopPolling();
          }
        } catch {
          // Process not found — likely already cleaned up
          setTeleState("error");
          setErrorMsg("Lost connection to process");
          setShowLogs(true);
          stopPolling();
        }
      }, 2000);
    },
    [stopPolling]
  );

  // Start polling if we already have a process running
  useEffect(() => {
    if (state.teleProcessId && teleState === "running") {
      startPolling(state.teleProcessId);
    }
    return stopPolling;
  }, [state.teleProcessId, teleState, startPolling, stopPolling]);

  async function handleStart() {
    setTeleState("starting");
    setErrorMsg(null);
    setShowLogs(false);
    try {
      await services.saveConfig(state);
      const res = await services.startTeleoperation(false);
      dispatch({ type: "SET_TELE_PROCESS_ID", id: res.process_id });
      setTeleState("running");
      startPolling(res.process_id);
    } catch (err) {
      setTeleState("error");
      setErrorMsg(err instanceof Error ? err.message : "Failed to start");
      setShowLogs(true);
    }
  }

  function handleStop() {
    if (!state.teleProcessId) return;
    stopPolling();
    services.stopProcess(state.teleProcessId).catch(() => {});
    dispatch({ type: "SET_TELE_PROCESS_ID", id: null });
    setTeleState("idle");
    setShowLogs(false);
    setErrorMsg(null);
  }

  function handleDismiss() {
    stopPolling();
    dispatch({ type: "SET_TELE_PROCESS_ID", id: null });
    setTeleState("idle");
    setShowLogs(false);
    setErrorMsg(null);
  }

  const isRunning = teleState === "running";
  const isError = teleState === "error";
  const isStarting = teleState === "starting";
  const { motors, motorOrder, frequency } = useMotorState(logs, isRunning);
  const summaryItems = buildSummary(state);

  return (
    <StepCard
      title="Teleoperation"
      description="Test your robot setup."
      showNext={false}
    >
      <div className="space-y-5">
        {!priorComplete && (
          <Alert>
            <AlertTriangle className="h-4 w-4" />
            <AlertDescription>
              Previous steps are not all completed. It is not recommended to
              proceed without completing them first.
            </AlertDescription>
          </Alert>
        )}

        {/* Config Summary */}
        <div className="space-y-2">
          <p className="text-sm font-medium">Configuration Summary</p>
          <div className="rounded-lg border bg-muted/50 p-4">
            {summaryItems.length > 0 ? (
              <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-sm">
                {summaryItems.map(({ label, value }) => (
                  <div key={label} className="contents">
                    <dt className="text-muted-foreground">{label}</dt>
                    <dd className="font-mono text-xs">{value}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="text-sm text-muted-foreground">
                Complete previous steps to see configuration.
              </p>
            )}
          </div>
        </div>

        <Separator />

        {/* Camera feed toggle — always available if cameras were selected */}
        {selectedCameraFeeds.length > 0 && (
          <div className="flex items-center gap-2">
            <Switch
              id="show-cameras"
              checked={showCameras}
              onCheckedChange={setShowCameras}
            />
            <Label htmlFor="show-cameras" className="text-sm cursor-pointer">
              Show camera feeds
            </Label>
          </div>
        )}

        {showCameras && <CameraFeedPanel cameras={selectedCameraFeeds} />}

        {/* Running state */}
        {isRunning && (
          <div className="space-y-3">
            <div className="flex items-center gap-3 rounded-lg border p-4">
              <CircleCheck className="h-5 w-5 text-muted-foreground shrink-0" />
              <div className="flex-1">
                <p className="text-sm font-medium">
                  Teleoperation is running
                </p>
                <p className="text-xs text-muted-foreground">
                  Move the leader arm to control the follower.
                </p>
              </div>
              <Button variant="outline" size="sm" onClick={handleStop}>
                <Square className="mr-2 h-3.5 w-3.5" />
                Stop
              </Button>
            </div>

            <MotorPanel motors={motors} motorOrder={motorOrder} frequency={frequency} />
          </div>
        )}

        {/* Error state */}
        {isError && (
          <div className="flex items-center gap-3 rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-950">
            <XCircle className="h-5 w-5 text-red-600 dark:text-red-400 shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-red-800 dark:text-red-200">
                Teleoperation failed
              </p>
              {errorMsg && (
                <p className="text-xs text-red-600 dark:text-red-400 mt-0.5">
                  {errorMsg}
                </p>
              )}
            </div>
            <Button variant="outline" size="sm" onClick={handleDismiss}>
              Dismiss
            </Button>
          </div>
        )}

        {/* Idle / Start button */}
        {!isRunning && !isError && (
          <Button
            onClick={handleStart}
            disabled={isStarting || !priorComplete}
          >
            {isStarting ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Play className="mr-2 h-4 w-4" />
            )}
            {isStarting ? "Starting..." : "Start Teleoperation"}
          </Button>
        )}

        {/* Collapsible Logs — always available when process exists */}
        {state.teleProcessId && (
          <div>
            <button
              type="button"
              onClick={() => setShowLogs(!showLogs)}
              className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {showLogs ? (
                <ChevronDown className="h-3.5 w-3.5" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5" />
              )}
              {showLogs ? "Hide Logs" : "Show Logs"}
              {logs.length > 0 && (
                <span className="text-muted-foreground/60">
                  ({logs.length} lines)
                </span>
              )}
            </button>
            {showLogs && (
              <div className="mt-2">
                <LogViewer
                  logs={logs}
                  isConnected={isConnected}
                  onClear={clearLogs}
                  maxHeight="300px"
                />
              </div>
            )}
          </div>
        )}
      </div>
    </StepCard>
  );
}

function buildSummary(
  state: ReturnType<typeof import("../wizard-provider").useWizard>["state"]
): { label: string; value: string }[] {
  const items: { label: string; value: string }[] = [];

  if (state.robotMode) {
    items.push({
      label: "Mode",
      value: state.robotMode === "bimanual" ? "Bimanual" : "Single Arm",
    });
  }

  for (const [role, port] of Object.entries(state.portAssignments)) {
    if (port) {
      items.push({
        label: role.replace(/_/g, " "),
        value: port.split(".").pop() || port,
      });
    }
  }

  const selectedCams = state.cameraSelections.filter((c) => c.included);
  if (selectedCams.length > 0) {
    items.push({
      label: "Cameras",
      value: selectedCams.map((c) => c.name).join(", "),
    });
  }

  for (const [role, file] of Object.entries(state.calibrationSelections)) {
    if (file) {
      items.push({
        label: `${role.replace(/_/g, " ")} cal`,
        value: file === "new" ? "New Calibration" : file,
      });
    }
  }

  return items;
}
