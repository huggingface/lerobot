"use client";

import { useState } from "react";
import { AlertTriangle, Loader2, Play, Square } from "lucide-react";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Separator } from "@/components/ui/separator";
import { Badge } from "@/components/ui/badge";
import { LogViewer } from "@/components/common/log-viewer";
import { useWebSocket } from "@/hooks/use-websocket";
import { services } from "@/lib/services";
import { useWizard } from "../wizard-provider";
import { StepCard } from "../step-card";

export function RecordStep() {
  const { state, dispatch, allPriorStepsComplete } = useWizard();
  const [starting, setStarting] = useState(false);
  const [stopping, setStopping] = useState(false);
  const priorComplete = allPriorStepsComplete(5);
  const isRunning = state.recordProcessId !== null;

  const { logs, isConnected, clearLogs } = useWebSocket(state.recordProcessId);

  const config = state.recordingConfig;
  const canStart =
    priorComplete &&
    config.repoId.trim() !== "" &&
    config.task.trim() !== "" &&
    config.numEpisodes > 0 &&
    config.episodeTimeS > 0;

  async function handleStart() {
    setStarting(true);
    try {
      const res = await services.startRecording(config);
      dispatch({ type: "SET_RECORD_PROCESS_ID", id: res.process_id });
    } finally {
      setStarting(false);
    }
  }

  async function handleStop() {
    if (!state.recordProcessId) return;
    setStopping(true);
    try {
      await services.stopRecording(state.recordProcessId);
      dispatch({ type: "SET_RECORD_PROCESS_ID", id: null });
    } finally {
      setStopping(false);
    }
  }

  function updateConfig(partial: Partial<typeof config>) {
    dispatch({ type: "SET_RECORDING_CONFIG", config: partial });
  }

  return (
    <StepCard title="Record Data" description="Configure and start data recording." showNext={false}>
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

        {/* Controls */}
        <div className="flex items-center gap-3">
          {!isRunning ? (
            <Button onClick={handleStart} disabled={starting || !canStart}>
              {starting ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Play className="mr-2 h-4 w-4" />
              )}
              Start Recording
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
              Recording
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
