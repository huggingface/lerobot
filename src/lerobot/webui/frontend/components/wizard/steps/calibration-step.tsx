"use client";

import { useEffect, useState } from "react";
import { Loader2, ImageIcon, Play, Square } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { getCalibrationPaths } from "@/lib/wizard-types";
import { services } from "@/lib/services";
import { useWebSocket } from "@/hooks/use-websocket";
import { LogViewer } from "@/components/common/log-viewer";
import { useWizard } from "../wizard-provider";
import { StepCard } from "../step-card";

const ROLE_LABELS: Record<string, string> = {
  follower: "Follower",
  leader: "Leader",
  left_follower: "Left Follower",
  right_follower: "Right Follower",
  left_leader: "Left Leader",
  right_leader: "Right Leader",
};

export function CalibrationStep() {
  const { state, dispatch } = useWizard();
  const [loading, setLoading] = useState(false);

  // Calibration process state (one at a time)
  const [calProcessId, setCalProcessId] = useState<string | null>(null);
  const [calibratingRole, setCalibratingRole] = useState<string | null>(null);
  const [calError, setCalError] = useState<string | null>(null);

  const { logs, isConnected, clearLogs } = useWebSocket(calProcessId);

  const calPaths = state.robotMode ? getCalibrationPaths(state.robotMode) : [];

  const allSelected = calPaths.every((p) => {
    const sel = state.calibrationSelections[p.role];
    if (sel === undefined || sel === null) return false;
    if (sel === "new")
      return (state.newCalibrationNames[p.role] || "").trim() !== "";
    return true;
  });

  // Load calibration files for each role on mount
  useEffect(() => {
    if (!state.robotMode) return;

    async function loadFiles() {
      setLoading(true);
      try {
        const paths = getCalibrationPaths(state.robotMode!);
        const seen = new Set<string>();
        for (const p of paths) {
          const key = `${p.category}/${p.robotType}`;
          if (seen.has(key)) continue;
          seen.add(key);
          const files = await services.listCalibrationFiles(
            p.category,
            p.robotType
          );
          dispatch({ type: "SET_CALIBRATION_FILES", key, files });
        }
      } finally {
        setLoading(false);
      }
    }

    loadFiles();
  }, [state.robotMode, dispatch]);

  async function handleStartCalibration(
    role: string,
    category: string,
    robotType: string
  ) {
    const name = (state.newCalibrationNames[role] || "").trim();
    if (!name) return;

    const port = state.portAssignments[role];
    if (!port) return;

    const deviceType = category === "robots" ? "robot" : "teleoperator";

    setCalError(null);
    clearLogs();
    setCalibratingRole(role);

    try {
      const res = await services.startCalibration(
        deviceType,
        name,
        robotType,
        port
      );
      setCalProcessId(res.process_id);
    } catch (err) {
      setCalError(err instanceof Error ? err.message : "Failed to start");
      setCalibratingRole(null);
    }
  }

  async function handleStopCalibration() {
    if (!calProcessId) return;
    try {
      await services.stopCalibration(calProcessId);
    } finally {
      setCalProcessId(null);
      setCalibratingRole(null);
    }
  }

  return (
    <StepCard
      title="Calibration"
      description="Choose an existing calibration file or start a new calibration for each arm."
      nextDisabled={!allSelected}
    >
      <div className="space-y-5">
        {loading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
            <span className="ml-2 text-sm text-muted-foreground">
              Loading calibration files...
            </span>
          </div>
        ) : (
          calPaths.map((p) => {
            const key = `${p.category}/${p.robotType}`;
            const files = state.calibrationFiles[key] || [];
            const selection = state.calibrationSelections[p.role];
            const isNew = selection === "new";
            const isCalibrating = calibratingRole === p.role;
            const port = state.portAssignments[p.role];

            return (
              <div key={p.role} className="space-y-1.5">
                <Label>
                  {ROLE_LABELS[p.role]}
                  <span className="ml-2 text-xs font-normal text-muted-foreground">
                    ({p.robotType})
                  </span>
                </Label>
                <Select
                  value={selection || ""}
                  onValueChange={(val) =>
                    dispatch({
                      type: "SET_CALIBRATION_SELECTION",
                      role: p.role,
                      filename: val,
                    })
                  }
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select calibration..." />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="new">
                      <span className="font-medium">+ New Calibration</span>
                    </SelectItem>
                    {files.map((file) => (
                      <SelectItem key={file} value={file}>
                        {file}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                {/* New Calibration panel */}
                {isNew && (
                  <div className="mt-2 space-y-4 rounded-lg border bg-muted/30 p-4">
                    {/* Name input */}
                    <div className="space-y-1.5">
                      <Label htmlFor={`cal-name-${p.role}`}>
                        Calibration File Name
                      </Label>
                      <Input
                        id={`cal-name-${p.role}`}
                        placeholder="e.g., left_follower"
                        value={state.newCalibrationNames[p.role] || ""}
                        onChange={(e) =>
                          dispatch({
                            type: "SET_NEW_CALIBRATION_NAME",
                            role: p.role,
                            name: e.target.value,
                          })
                        }
                      />
                      <p className="text-xs text-muted-foreground">
                        This name will be used as the robot/teleoperator ID for
                        teleoperation and recording.
                      </p>
                    </div>

                    {/* Image placeholder */}
                    <div className="flex items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/25 bg-muted/50 p-8">
                      <div className="text-center text-sm text-muted-foreground">
                        <ImageIcon className="mx-auto mb-2 h-10 w-10 text-muted-foreground/50" />
                        <p>
                          Place the robot in the calibration position shown
                          below before starting.
                        </p>
                        <p className="mt-1 text-xs italic">
                          (Reference image coming soon)
                        </p>
                      </div>
                    </div>

                    {/* Start / Stop button */}
                    {isCalibrating && calProcessId ? (
                      <Button
                        variant="destructive"
                        className="w-full"
                        onClick={handleStopCalibration}
                      >
                        <Square className="mr-2 h-4 w-4" />
                        Stop Calibration
                      </Button>
                    ) : (
                      <Button
                        className="w-full"
                        disabled={
                          !(state.newCalibrationNames[p.role] || "").trim() ||
                          !port ||
                          (calibratingRole !== null &&
                            calibratingRole !== p.role)
                        }
                        onClick={() =>
                          handleStartCalibration(
                            p.role,
                            p.category,
                            p.robotType
                          )
                        }
                      >
                        <Play className="mr-2 h-4 w-4" />
                        Start Calibration
                      </Button>
                    )}

                    {!port && (
                      <p className="text-xs text-destructive">
                        No port assigned for this role. Go back to the Ports
                        step to assign one.
                      </p>
                    )}

                    {calError && isCalibrating && (
                      <p className="text-xs text-destructive">{calError}</p>
                    )}

                    {/* Log viewer when calibrating this role */}
                    {isCalibrating && calProcessId && (
                      <LogViewer
                        logs={logs}
                        isConnected={isConnected}
                        onClear={clearLogs}
                        maxHeight="250px"
                      />
                    )}
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </StepCard>
  );
}
