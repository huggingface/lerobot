"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
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
import { useMotorState, MotorPanel, CameraFeedPanel } from "@/components/common/robot-display";
import { useWebSocket } from "@/hooks/use-websocket";
import { services } from "@/lib/services";
import { useWizard } from "../wizard-provider";
import { StepCard } from "../step-card";

type TeleState = "idle" | "starting" | "running" | "error" | "stopped";

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
