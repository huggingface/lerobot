"use client";

import { useEffect, useState } from "react";
import {
  Loader2,
  ImageIcon,
  Check,
  CircleDot,
  Circle,
  AlertCircle,
} from "lucide-react";
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
import {
  useManualCalibration,
  type MotorValues,
} from "@/hooks/use-manual-calibration";
import { DevErrorPanel } from "@/components/common/dev-error-panel";
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

// ─── Encoder table component ────────────────────────────────────────────────

function EncoderTable({
  motors,
  positions,
}: {
  motors: string[];
  positions: Record<string, MotorValues>;
}) {
  return (
    <div className="overflow-x-auto rounded-lg border bg-black/80">
      <table className="w-full text-xs font-mono">
        <thead>
          <tr className="border-b border-white/10 text-green-400/70">
            <th className="px-3 py-2 text-left font-medium">Motor</th>
            <th className="px-3 py-2 text-right font-medium">Min</th>
            <th className="px-3 py-2 text-right font-medium">Current</th>
            <th className="px-3 py-2 text-right font-medium">Max</th>
          </tr>
        </thead>
        <tbody className="text-green-400">
          {motors.map((motor) => {
            const v = positions[motor];
            return (
              <tr key={motor} className="border-b border-white/5 last:border-0">
                <td className="px-3 py-1.5 text-green-300">{motor}</td>
                <td className="px-3 py-1.5 text-right tabular-nums">
                  {v?.min ?? "—"}
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums font-bold">
                  {v?.pos ?? "—"}
                </td>
                <td className="px-3 py-1.5 text-right tabular-nums">
                  {v?.max ?? "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ─── Phase indicator ─────────────────────────────────────────────────────────

function PhaseIndicator({ phase }: { phase: number }) {
  const steps = ["Set Middle Position", "Record Range of Motion"];
  return (
    <div className="flex items-center gap-3">
      {steps.map((label, i) => {
        const done = i < phase;
        const active = i === phase;
        return (
          <div key={label} className="flex items-center gap-1.5 text-xs">
            {done ? (
              <Check className="h-3.5 w-3.5 text-green-500" />
            ) : active ? (
              <CircleDot className="h-3.5 w-3.5 text-primary" />
            ) : (
              <Circle className="h-3.5 w-3.5 text-muted-foreground/40" />
            )}
            <span
              className={
                active
                  ? "font-medium text-foreground"
                  : done
                    ? "text-green-600 dark:text-green-400"
                    : "text-muted-foreground/60"
              }
            >
              {label}
            </span>
            {i < steps.length - 1 && (
              <div className="ml-1 h-px w-6 bg-muted-foreground/20" />
            )}
          </div>
        );
      })}
    </div>
  );
}

// ─── New calibration panel ───────────────────────────────────────────────────

function NewCalibrationPanel({
  role,
  category,
  robotType,
}: {
  role: string;
  category: string;
  robotType: string;
}) {
  const { state: wizState, dispatch } = useWizard();
  const cal = useManualCalibration();
  const { state: calState } = cal;

  const name = (wizState.newCalibrationNames[role] || "").trim();
  const port = wizState.portAssignments[role];

  const calPhase = calState.phase;
  const isIdle =
    calPhase === "disconnected" || calPhase === "error" || calPhase === "saved";

  // Determine which step we're in (0 = homing, 1 = recording)
  const uiPhase =
    calPhase === "homing_done" ||
    calPhase === "recording" ||
    calPhase === "saving" ||
    calPhase === "saved"
      ? 1
      : 0;

  function handleConnect() {
    if (!name || !port) return;
    const deviceType = category === "robots" ? "robot" : "teleoperator";
    cal.connect(port, deviceType, robotType, name);
  }

  function handleSetHoming() {
    cal.setHoming();
  }

  function handleStartRecording() {
    cal.startRecording();
  }

  function handleDone() {
    cal.stopAndSave();
  }

  function handleReset() {
    cal.reset();
  }

  return (
    <div className="mt-2 space-y-4 rounded-lg border bg-muted/30 p-4">
      {/* Name input */}
      <div className="space-y-1.5">
        <Label htmlFor={`cal-name-${role}`}>Calibration File Name</Label>
        <Input
          id={`cal-name-${role}`}
          placeholder="e.g., left_follower"
          value={wizState.newCalibrationNames[role] || ""}
          onChange={(e) =>
            dispatch({
              type: "SET_NEW_CALIBRATION_NAME",
              role,
              name: e.target.value,
            })
          }
          disabled={!isIdle}
        />
        <p className="text-xs text-muted-foreground">
          This name will be used as the robot/teleoperator ID for teleoperation
          and recording.
        </p>
      </div>

      {!port && (
        <p className="text-xs text-destructive">
          No port assigned for this role. Go back to the Ports step to assign
          one.
        </p>
      )}

      {/* Show phase indicator once connected */}
      {!isIdle && calPhase !== "connecting" && <PhaseIndicator phase={uiPhase} />}

      {/* ── Step 1: Homing ── */}
      {(calPhase === "disconnected" ||
        calPhase === "connecting" ||
        calPhase === "connected" ||
        calPhase === "homing" ||
        calPhase === "error") && (
        <>
          {/* Reference image placeholder */}
          <div className="flex items-center justify-center rounded-lg border-2 border-dashed border-muted-foreground/25 bg-muted/50 p-8">
            <div className="text-center text-sm text-muted-foreground">
              <ImageIcon className="mx-auto mb-2 h-10 w-10 text-muted-foreground/50" />
              <p>
                Move the arm to the <strong>middle</strong> of its range of
                motion before proceeding.
              </p>
              <p className="mt-1 text-xs italic">
                (Reference image coming soon)
              </p>
            </div>
          </div>

          {/* Connect + Set Homing button */}
          {calPhase === "disconnected" || calPhase === "error" ? (
            <Button
              className="w-full"
              disabled={!name || !port}
              onClick={handleConnect}
            >
              Connect &amp; Set Middle Position
            </Button>
          ) : calPhase === "connecting" ? (
            <Button className="w-full" disabled>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Connecting...
            </Button>
          ) : calPhase === "connected" ? (
            <Button className="w-full" onClick={handleSetHoming}>
              Set Middle Position
            </Button>
          ) : calPhase === "homing" ? (
            <Button className="w-full" disabled>
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              Setting homing offsets...
            </Button>
          ) : null}
        </>
      )}

      {/* ── Step 2: Range Recording ── */}
      {(calPhase === "homing_done" || calPhase === "recording") && (
        <>
          <div className="space-y-2">
            <p className="text-sm text-muted-foreground">
              Move all joints sequentially through their{" "}
              <strong>entire range of motion</strong>. The table below updates
              live. Click <strong>Done</strong> when finished.
            </p>

            <EncoderTable
              motors={calState.motors}
              positions={calState.positions}
            />
          </div>

          {calPhase === "homing_done" ? (
            <Button className="w-full" onClick={handleStartRecording}>
              Start Recording
            </Button>
          ) : (
            <Button className="w-full" onClick={handleDone}>
              <Check className="mr-2 h-4 w-4" />
              Done
            </Button>
          )}
        </>
      )}

      {/* ── Saving ── */}
      {calPhase === "saving" && (
        <div className="flex items-center justify-center py-4">
          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          <span className="text-sm text-muted-foreground">
            Saving calibration...
          </span>
        </div>
      )}

      {/* ── Saved ── */}
      {calPhase === "saved" && (
        <div className="space-y-3">
          <div className="flex items-center gap-2 rounded-lg border border-green-200 bg-green-50 p-3 dark:border-green-900 dark:bg-green-950">
            <Check className="h-5 w-5 text-green-600 dark:text-green-400" />
            <div>
              <p className="text-sm font-medium text-green-800 dark:text-green-200">
                Calibration saved successfully
              </p>
              {calState.savedPath && (
                <p className="text-xs text-green-600 dark:text-green-400 font-mono break-all">
                  {calState.savedPath}
                </p>
              )}
            </div>
          </div>
          <Button variant="outline" className="w-full" onClick={handleReset}>
            Calibrate Again
          </Button>
        </div>
      )}

      {/* ── Error ── */}
      {calState.error && (
        <DevErrorPanel error={new Error(calState.error)} />
      )}
    </div>
  );
}

// ─── Main step component ─────────────────────────────────────────────────────

export function CalibrationStep() {
  const { state, dispatch } = useWizard();
  const [loading, setLoading] = useState(false);

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

                {isNew && (
                  <NewCalibrationPanel
                    role={p.role}
                    category={p.category}
                    robotType={p.robotType}
                  />
                )}
              </div>
            );
          })
        )}
      </div>
    </StepCard>
  );
}
