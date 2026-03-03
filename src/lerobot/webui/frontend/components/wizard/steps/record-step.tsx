"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clapperboard,
  Film,
  FolderOpen,
  Loader2,
  Play,
  RefreshCw,
  Square,
  XCircle,
} from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { LogViewer } from "@/components/common/log-viewer";
import {
  useMotorState,
  MotorPanel,
  CameraFeedPanel,
} from "@/components/common/robot-display";
import { useWebSocket } from "@/hooks/use-websocket";
import { services } from "@/lib/services";
import { useWizard } from "../wizard-provider";
import { StepCard } from "../step-card";

// ---------------------------------------------------------------------------
// Recording phase detection
// ---------------------------------------------------------------------------

type RecordPhase = "idle" | "recording" | "resetting" | "encoding" | "done";

const EPISODE_RE = /Recording episode (\d+)/;
const RESET_RE = /Reset the environment/;
const ENCODING_RE = /[Ee]ncoding (?:video|videos|remaining)/;
const DONE_RE = /Stop recording|Exiting/;

function useRecordingPhase(logs: string[], isRunning: boolean) {
  const [phase, setPhase] = useState<RecordPhase>("idle");
  const [currentEpisode, setCurrentEpisode] = useState<number | null>(null);

  useEffect(() => {
    if (!isRunning) {
      setPhase("idle");
      return;
    }

    const tail = logs.slice(-30);
    for (let i = tail.length - 1; i >= 0; i--) {
      const line = tail[i];

      if (DONE_RE.test(line)) {
        setPhase("done");
        return;
      }
      if (ENCODING_RE.test(line)) {
        setPhase("encoding");
        return;
      }
      if (RESET_RE.test(line)) {
        setPhase("resetting");
        return;
      }
      const epMatch = line.match(EPISODE_RE);
      if (epMatch) {
        setPhase("recording");
        setCurrentEpisode(parseInt(epMatch[1], 10));
        return;
      }
    }
  }, [logs, isRunning]);

  return { phase, currentEpisode };
}

// ---------------------------------------------------------------------------
// Repo ID duplicate tracking via localStorage
// ---------------------------------------------------------------------------

const STORAGE_KEY = "lerobot_saved_repo_ids";

function getSavedRepoIds(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? (JSON.parse(raw) as string[]) : [];
  } catch {
    return [];
  }
}

function saveRepoId(repoId: string) {
  const existing = getSavedRepoIds();
  if (!existing.includes(repoId)) {
    localStorage.setItem(STORAGE_KEY, JSON.stringify([...existing, repoId]));
  }
}

// ---------------------------------------------------------------------------
// Phase status card
// ---------------------------------------------------------------------------

function RecordingStatusCard({
  phase,
  currentEpisode,
  numEpisodes,
  onStop,
  stopping,
}: {
  phase: RecordPhase;
  currentEpisode: number | null;
  numEpisodes: number;
  onStop: () => void;
  stopping: boolean;
}) {
  if (phase === "idle") return null;

  type PhaseConfig = {
    icon: React.ReactNode;
    title: string;
    subtitle: string;
    accent: string;
  };

  const configs: Record<RecordPhase, PhaseConfig> = {
    idle: { icon: null, title: "", subtitle: "", accent: "" },
    recording: {
      icon: <Clapperboard className="h-5 w-5 text-red-500 shrink-0" />,
      title:
        currentEpisode !== null
          ? `Recording episode ${currentEpisode} / ${numEpisodes}`
          : "Recording…",
      subtitle: "Perform the task now.",
      accent: "border-red-200 dark:border-red-900",
    },
    resetting: {
      icon: <RefreshCw className="h-5 w-5 text-amber-500 shrink-0" />,
      title: "Reset the environment",
      subtitle: "Prepare the scene for the next episode.",
      accent: "border-amber-200 dark:border-amber-900",
    },
    encoding: {
      icon: <Film className="h-5 w-5 text-blue-500 shrink-0" />,
      title: "Encoding videos…",
      subtitle: "Please wait while the episode is being saved.",
      accent: "border-blue-200 dark:border-blue-900",
    },
    done: {
      icon: <CheckCircle2 className="h-5 w-5 text-emerald-500 shrink-0" />,
      title: "All episodes complete",
      subtitle: "The dataset is being finalised and uploaded.",
      accent: "border-emerald-200 dark:border-emerald-900",
    },
  };

  const cfg = configs[phase];

  return (
    <div className={`flex items-center gap-3 rounded-lg border p-4 ${cfg.accent}`}>
      {cfg.icon}
      <div className="flex-1">
        <p className="text-sm font-medium">{cfg.title}</p>
        <p className="text-xs text-muted-foreground mt-0.5">{cfg.subtitle}</p>
      </div>
      {phase !== "done" && (
        <Button variant="outline" size="sm" onClick={onStop} disabled={stopping}>
          {stopping ? (
            <Loader2 className="mr-2 h-3.5 w-3.5 animate-spin" />
          ) : (
            <Square className="mr-2 h-3.5 w-3.5" />
          )}
          Stop
        </Button>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export function RecordStep() {
  const { state, dispatch, allPriorStepsComplete } = useWizard();
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const [showLogs, setShowLogs] = useState(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [repoIdWarning, setRepoIdWarning] = useState(false);
  const priorComplete = allPriorStepsComplete(5);
  const isRunning = state.recordProcessId !== null;

  const { logs, isConnected, clearLogs } = useWebSocket(state.recordProcessId);

  const config = state.recordingConfig;

  // Phase detection from log stream
  const { phase, currentEpisode } = useRecordingPhase(logs, isRunning);

  // Motor + camera feeds (only when displayData is on)
  const { motors, motorOrder, frequency } = useMotorState(
    logs,
    isRunning && config.displayData
  );
  const selectedCameraFeeds = state.cameraSelections
    .filter((c) => c.included && c.name)
    .map((c) => ({ deviceId: c.deviceId, name: c.name }));

  // Process crash polling
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
            setErrorMsg(status.error_message || "Process exited with an error");
            setShowLogs(true);
            dispatch({ type: "SET_RECORD_PROCESS_ID", id: null });
            stopPolling();
          } else if (status.state === "stopped") {
            dispatch({ type: "SET_RECORD_PROCESS_ID", id: null });
            stopPolling();
          }
        } catch {
          setErrorMsg("Lost connection to process");
          setShowLogs(true);
          dispatch({ type: "SET_RECORD_PROCESS_ID", id: null });
          stopPolling();
        }
      }, 2000);
    },
    [stopPolling, dispatch]
  );

  // Resume polling if process was already running when component mounts
  useEffect(() => {
    if (state.recordProcessId) {
      startPolling(state.recordProcessId);
    }
    return stopPolling;
  }, [state.recordProcessId, startPolling, stopPolling]);

  // Check if repo ID was used before
  useEffect(() => {
    if (config.repoId.trim() === "") {
      setRepoIdWarning(false);
      return;
    }
    setRepoIdWarning(getSavedRepoIds().includes(config.repoId.trim()));
  }, [config.repoId]);

  const canStart =
    priorComplete &&
    config.repoId.trim() !== "" &&
    config.task.trim() !== "" &&
    config.numEpisodes > 0 &&
    config.episodeTimeS > 0;

  async function handleStart() {
    setStarting(true);
    setErrorMsg(null);
    setShowLogs(false);
    try {
      // Clear cached data so the new recording replaces any previous dataset
      if (config.repoId.trim()) {
        await services.clearCache(config.repoId.trim()).catch(() => {});
      }
      const res = await services.startRecording(config);
      dispatch({ type: "SET_RECORD_PROCESS_ID", id: res.process_id });
      saveRepoId(config.repoId.trim());
      setRepoIdWarning(false);
      startPolling(res.process_id);
    } catch (err) {
      setErrorMsg(err instanceof Error ? err.message : "Failed to start recording");
      setShowLogs(true);
    } finally {
      setStarting(false);
    }
  }

  async function handleStop() {
    if (!state.recordProcessId) return;
    setStopping(true);
    stopPolling();
    try {
      await services.stopRecording(state.recordProcessId);
    } finally {
      dispatch({ type: "SET_RECORD_PROCESS_ID", id: null });
      setStopping(false);
    }
  }

  function updateConfig(partial: Partial<typeof config>) {
    dispatch({ type: "SET_RECORDING_CONFIG", config: partial });
  }

  // Show the log toggle once a process has produced any output
  const hasLogs = logs.length > 0;

  return (
    <StepCard
      title="Record Data"
      description="Configure and start data recording."
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

        {/* Form fields */}
        <div className="space-y-4">
          <div className="space-y-1.5">
            <Label htmlFor="repo-id">HuggingFace Repo ID</Label>
            <Input
              id="repo-id"
              placeholder="username/dataset_name"
              value={config.repoId}
              onChange={(e) => updateConfig({ repoId: e.target.value })}
              disabled={isRunning}
            />
            {repoIdWarning && (
              <p className="flex items-center gap-1.5 text-xs text-amber-600 dark:text-amber-400">
                <AlertTriangle className="h-3.5 w-3.5 shrink-0" />
                A dataset with this repo ID was recorded before. Starting will
                overwrite the existing dataset.
              </p>
            )}
          </div>

          <div className="space-y-1.5">
            <Label htmlFor="task">Task Description</Label>
            <Input
              id="task"
              placeholder="e.g. Pick up the cube and place it in the bin"
              value={config.task}
              onChange={(e) => updateConfig({ task: e.target.value })}
              disabled={isRunning}
            />
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-1.5">
              <Label htmlFor="num-episodes">Episodes</Label>
              <Input
                id="num-episodes"
                type="number"
                min={1}
                value={config.numEpisodes}
                onChange={(e) =>
                  updateConfig({ numEpisodes: parseInt(e.target.value) || 1 })
                }
                disabled={isRunning}
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="episode-time">Episode Time (s)</Label>
              <Input
                id="episode-time"
                type="number"
                min={1}
                value={config.episodeTimeS}
                onChange={(e) =>
                  updateConfig({ episodeTimeS: parseInt(e.target.value) || 1 })
                }
                disabled={isRunning}
              />
            </div>

            <div className="space-y-1.5">
              <Label htmlFor="reset-time">Reset Time (s)</Label>
              <Input
                id="reset-time"
                type="number"
                min={0}
                value={config.resetTimeS}
                onChange={(e) =>
                  updateConfig({ resetTimeS: parseInt(e.target.value) || 0 })
                }
                disabled={isRunning}
              />
            </div>
          </div>

          <div className="flex items-center gap-3">
            <Switch
              id="display-data"
              checked={config.displayData}
              onCheckedChange={(checked) =>
                updateConfig({ displayData: checked })
              }
              disabled={isRunning}
            />
            <Label htmlFor="display-data">Display data while recording</Label>
          </div>
        </div>

        <Separator />

        {/* Start button + folder button */}
        <div className="flex items-center gap-2">
          {!isRunning && !errorMsg && (
            <Button onClick={handleStart} disabled={starting || !canStart}>
              {starting ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Play className="mr-2 h-4 w-4" />
              )}
              {starting ? "Starting…" : "Start Recording"}
            </Button>
          )}
          <Button
            variant="outline"
            size="icon"
            title="Open data folder (~/.cache/huggingface/lerobot)"
            onClick={() => services.openDataFolder().catch(() => {})}
          >
            <FolderOpen className="h-4 w-4" />
          </Button>
        </div>

        {/* Error state */}
        {errorMsg && (
          <div className="flex items-center gap-3 rounded-lg border border-red-200 bg-red-50 p-4 dark:border-red-900 dark:bg-red-950">
            <XCircle className="h-5 w-5 text-red-600 dark:text-red-400 shrink-0" />
            <div className="flex-1">
              <p className="text-sm font-medium text-red-800 dark:text-red-200">
                Recording failed
              </p>
              <p className="text-xs text-red-600 dark:text-red-400 mt-0.5">
                {errorMsg}
              </p>
            </div>
            <Button
              variant="outline"
              size="sm"
              onClick={() => {
                setErrorMsg(null);
                setShowLogs(false);
              }}
            >
              Dismiss
            </Button>
          </div>
        )}

        {/* Phase status banner (shown while process is running) */}
        {isRunning && (
          <RecordingStatusCard
            phase={phase}
            currentEpisode={currentEpisode}
            numEpisodes={config.numEpisodes}
            onStop={handleStop}
            stopping={stopping}
          />
        )}

        {/* Live camera + motor feeds (only when displayData is enabled) */}
        {isRunning && config.displayData && (
          <div className="space-y-3">
            {selectedCameraFeeds.length > 0 && (
              <CameraFeedPanel cameras={selectedCameraFeeds} />
            )}
            <MotorPanel
              motors={motors}
              motorOrder={motorOrder}
              frequency={frequency}
            />
          </div>
        )}

        {/* Collapsible terminal logs */}
        {hasLogs && (
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
              <span className="text-muted-foreground/60">({logs.length} lines)</span>
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
