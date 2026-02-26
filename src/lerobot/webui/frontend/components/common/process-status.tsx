"use client";

import { Badge } from "@/components/ui/badge";

type ProcessState = "not_started" | "running" | "stopped" | "error";

interface ProcessStatusBadgeProps {
  state: ProcessState;
  pid?: number | null;
  uptimeSeconds?: number | null;
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m ${s}s`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export function ProcessStatusBadge({
  state,
  pid,
  uptimeSeconds,
}: ProcessStatusBadgeProps) {
  const variants: Record<ProcessState, string> = {
    not_started: "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400",
    running:
      "bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400",
    stopped:
      "bg-zinc-100 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400",
    error: "bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400",
  };

  const labels: Record<ProcessState, string> = {
    not_started: "Not Started",
    running: "Running",
    stopped: "Stopped",
    error: "Error",
  };

  return (
    <div className="flex items-center gap-2">
      <Badge variant="outline" className={variants[state]}>
        {state === "running" && (
          <span className="mr-1.5 inline-block h-1.5 w-1.5 animate-pulse rounded-full bg-green-500" />
        )}
        {labels[state]}
      </Badge>
      {pid && (
        <span className="text-xs text-muted-foreground">PID: {pid}</span>
      )}
      {uptimeSeconds != null && state === "running" && (
        <span className="text-xs text-muted-foreground">
          {formatUptime(uptimeSeconds)}
        </span>
      )}
    </div>
  );
}
