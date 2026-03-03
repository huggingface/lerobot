"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Camera } from "lucide-react";

// Regex to match motor position lines like "shoulder_pan.pos        |  12.45"
const MOTOR_LINE_RE = /^([\w.]+)\s+\|\s+([-+]?\d+\.\d+)\s*$/;
// Regex to match timing lines like "time: 16.67ms (60 Hz)"
const TIME_LINE_RE = /^time:\s+([\d.]+)ms\s+\((\d+)\s+Hz\)/;

const MOTOR_RANGE = 100; // Normalized motor positions are roughly in [-100, 100]

/** Stable motor state: only updates at a throttled rate to avoid glitching. */
export function useMotorState(logs: string[], isRunning: boolean) {
  const [motors, setMotors] = useState<Record<string, number>>({});
  const [motorOrder, setMotorOrder] = useState<string[]>([]);
  const [frequency, setFrequency] = useState<number | null>(null);
  const lastUpdateRef = useRef(0);

  useEffect(() => {
    if (!isRunning) return;

    const now = performance.now();
    if (now - lastUpdateRef.current < 80) return; // Throttle to ~12Hz
    lastUpdateRef.current = now;

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
      setMotorOrder((prev) =>
        prev.length > 0 ? prev : Object.keys(parsed)
      );
    }
    if (freq !== null) setFrequency(freq);
  }, [logs, isRunning]);

  return { motors, motorOrder, frequency };
}

/** Single motor row with a visual position bar and numeric value. */
function MotorRow({ name, value }: { name: string; value: number }) {
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
        <div className="absolute left-1/2 top-0 bottom-0 w-px bg-muted-foreground/20" />
        {value >= 0 ? (
          <div
            className="absolute top-0 bottom-0 left-1/2 rounded-r-full bg-emerald-500/80 transition-all duration-75"
            style={{ width: `${pct - 50}%` }}
          />
        ) : (
          <div
            className="absolute top-0 bottom-0 right-1/2 rounded-l-full bg-emerald-500/80 transition-all duration-75"
            style={{ width: `${50 - pct}%` }}
          />
        )}
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
export function MotorPanel({
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

/** Single camera feed using the browser getUserMedia API. */
export function CameraFeed({ deviceId }: { deviceId: string }) {
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

/** Live camera feed panel showing the cameras passed in. */
export function CameraFeedPanel({
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
