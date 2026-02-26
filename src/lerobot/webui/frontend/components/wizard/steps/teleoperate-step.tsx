"use client";

import { useState } from "react";
import { AlertTriangle, Loader2, Play, Square } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { LogViewer } from "@/components/common/log-viewer";
import { useWebSocket } from "@/hooks/use-websocket";
import { services } from "@/lib/services";
import { useWizard } from "../wizard-provider";
import { StepCard } from "../step-card";

export function TeleoperateStep() {
  const { state, dispatch, allPriorStepsComplete } = useWizard();
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const priorComplete = allPriorStepsComplete(4);
  const isRunning = state.teleProcessId !== null;

  const { logs, isConnected, clearLogs } = useWebSocket(state.teleProcessId);

  async function handleStart() {
    setStarting(true);
    try {
      const res = await services.startTeleoperation(true);
      dispatch({ type: "SET_TELE_PROCESS_ID", id: res.process_id });
    } finally {
      setStarting(false);
    }
  }

  async function handleStop() {
    if (!state.teleProcessId) return;
    setStopping(true);
    try {
      await services.stopProcess(state.teleProcessId);
      dispatch({ type: "SET_TELE_PROCESS_ID", id: null });
    } finally {
      setStopping(false);
    }
  }

  // Build summary items from wizard state
  const summaryItems = buildSummary(state);

  return (
    <StepCard title="Teleoperation" description="Test your robot setup." showNext={false}>
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

        {/* Controls */}
        <div className="flex items-center gap-3">
          {!isRunning ? (
            <Button onClick={handleStart} disabled={starting || !priorComplete}>
              {starting ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Play className="mr-2 h-4 w-4" />
              )}
              Start Teleoperation
            </Button>
          ) : (
            <Button
              variant="destructive"
              onClick={handleStop}
              disabled={stopping}
            >
              {stopping ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Square className="mr-2 h-4 w-4" />
              )}
              Stop
            </Button>
          )}
          {isRunning && (
            <Badge variant="outline" className="text-green-600">
              Running
            </Badge>
          )}
        </div>

        {/* Logs */}
        {isRunning && (
          <LogViewer
            logs={logs}
            isConnected={isConnected}
            onClear={clearLogs}
          />
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
