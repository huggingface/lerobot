"use client";

/**
 * LeKiwi 三轮全向底盘键盘控制 UI
 * 用于 XLeRobot-SmolVLA webui，与后端 /move/teleop、/move/init 配合。
 * 按键：WASD/方向键 平移，Q/E 旋转；支持 direction_x, direction_y, direction_theta。
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { Navigation, Play, Square } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AlertCircle } from "lucide-react";

// LeKiwi 3-wheel: direction_x left/right, direction_y forward/back, direction_theta rotate
export const CHASSIS_KEY_MAPPINGS: Record<
  string,
  { direction_x: number; direction_y: number; direction_theta: number }
> = {
  ArrowUp: { direction_x: 0, direction_y: 1, direction_theta: 0 },
  ArrowDown: { direction_x: 0, direction_y: -1, direction_theta: 0 },
  ArrowLeft: { direction_x: -1, direction_y: 0, direction_theta: 0 },
  ArrowRight: { direction_x: 1, direction_y: 0, direction_theta: 0 },
  w: { direction_x: 0, direction_y: 1, direction_theta: 0 },
  s: { direction_x: 0, direction_y: -1, direction_theta: 0 },
  a: { direction_x: -1, direction_y: 0, direction_theta: 0 },
  d: { direction_x: 1, direction_y: 0, direction_theta: 0 },
  q: { direction_x: 0, direction_y: 0, direction_theta: 1 },
  e: { direction_x: 0, direction_y: 0, direction_theta: -1 },
};

export interface RobotStatusItem {
  name: string;
  device_name: string;
  robot_type: string;
}

export interface LeKiwiChassisControlProps {
  /** 当前连接的机器人列表（含 mobile 类型） */
  connectedRobots: RobotStatusItem[];
  /** API 根地址，如 "http://localhost:8080" */
  apiBaseUrl?: string;
  /** 自定义请求函数；不传则用 fetch(apiBaseUrl + url, { method, body }) */
  fetchApi?: (
    url: string,
    method: string,
    body?: object
  ) => Promise<unknown>;
  /** 可选：toast 回调 */
  onToast?: (type: "success" | "error" | "info", message: string) => void;
}

export function LeKiwiChassisControl({
  connectedRobots,
  apiBaseUrl = "",
  fetchApi,
  onToast,
}: LeKiwiChassisControlProps) {
  const [selectedChassisId, setSelectedChassisId] = useState<number | null>(
    null
  );
  const [isChassisActive, setIsChassisActive] = useState(false);
  const [activeChassisKey, setActiveChassisKey] = useState<string | null>(null);
  const [chassisSpeed, setChassisSpeed] = useState(0.5);
  const chassisKeysRef = useRef(new Set<string>());
  const chassisLoopRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const mobileRobots = connectedRobots.filter((r) => r.robot_type === "mobile");

  const request = useCallback(
    async (url: string, method: string, body?: object) => {
      if (fetchApi) return fetchApi(url, method, body);
      const res = await fetch(`${apiBaseUrl}${url}`, {
        method,
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      });
      if (!res.ok) throw new Error(res.statusText);
      return res.json?.() ?? undefined;
    },
    [apiBaseUrl, fetchApi]
  );

  const sendChassisCommand = useCallback(
    async (dirX: number, dirY: number, dirTheta: number = 0) => {
      if (selectedChassisId === null) return;
      const controlData = {
        source: "right",
        direction_x: dirX * chassisSpeed,
        direction_y: dirY * chassisSpeed,
        direction_theta: dirTheta * chassisSpeed,
        x: 0,
        y: 0,
        z: 0,
        rx: 0,
        ry: 0,
        rz: 0,
        open: 1,
        timestamp: Date.now() / 1000,
      };
      try {
        await request(
          `/move/teleop?robot_id=${selectedChassisId}`,
          "POST",
          controlData
        );
      } catch {
        // 静默忽略发送失败
      }
    },
    [selectedChassisId, chassisSpeed, request]
  );

  const handleChassisStart = useCallback(async () => {
    if (selectedChassisId === null) {
      onToast?.("error", "Please select a chassis robot first");
      return;
    }
    await request(`/move/init?robot_id=${selectedChassisId}`, "POST", {});
    setIsChassisActive(true);
    onToast?.("success", "Chassis keyboard control started (WASD / Q E)");
  }, [selectedChassisId, request, onToast]);

  const handleChassisStop = useCallback(() => {
    setIsChassisActive(false);
    chassisKeysRef.current.clear();
    setActiveChassisKey(null);
    sendChassisCommand(0, 0, 0);
    onToast?.("info", "Chassis control stopped");
  }, [sendChassisCommand, onToast]);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (!isChassisActive || e.repeat) return;
      const key = e.key;
      const keyLower = key.toLowerCase();
      if (CHASSIS_KEY_MAPPINGS[key] ?? CHASSIS_KEY_MAPPINGS[keyLower]) {
        e.preventDefault();
        setActiveChassisKey(key);
        chassisKeysRef.current.add(
          CHASSIS_KEY_MAPPINGS[key] ? key : keyLower
        );
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      chassisKeysRef.current.delete(e.key);
      chassisKeysRef.current.delete(e.key.toLowerCase());
      if (chassisKeysRef.current.size === 0) setActiveChassisKey(null);
    };
    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
    };
  }, [isChassisActive]);

  useEffect(() => {
    const loop = () => {
      if (!isChassisActive) return;
      let dx = 0,
        dy = 0,
        dtheta = 0;
      chassisKeysRef.current.forEach((k) => {
        const m =
          CHASSIS_KEY_MAPPINGS[k.toLowerCase()] ?? CHASSIS_KEY_MAPPINGS[k];
        if (m) {
          dx += m.direction_x;
          dy += m.direction_y;
          dtheta += m.direction_theta;
        }
      });
      dx = Math.max(-1, Math.min(1, dx));
      dy = Math.max(-1, Math.min(1, dy));
      dtheta = Math.max(-1, Math.min(1, dtheta));
      sendChassisCommand(dx, dy, dtheta);
    };
    chassisLoopRef.current = setInterval(loop, 50);
    return () => {
      if (chassisLoopRef.current) clearInterval(chassisLoopRef.current);
    };
  }, [isChassisActive, sendChassisCommand]);

  useEffect(() => {
    return () => {
      if (isChassisActive) {
        chassisKeysRef.current.clear();
        sendChassisCommand(0, 0, 0);
      }
    };
  }, [isChassisActive, sendChassisCommand]);

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle className="flex items-center gap-x-2">
          <Navigation className="size-4" />
          LeKiwi Chassis Control
        </CardTitle>
        <CardDescription>
          Use WASD or arrow keys to move, Q/E to rotate (LeKiwi 3-wheel). Works
          with backend smooth acceleration.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-wrap gap-4 items-end">
          <div className="space-y-2">
            <Label htmlFor="chassis-select">Chassis Robot</Label>
            <Select
              value={selectedChassisId?.toString() ?? ""}
              onValueChange={(v) => setSelectedChassisId(parseInt(v))}
              disabled={isChassisActive}
            >
              <SelectTrigger id="chassis-select" className="w-[220px]">
                <SelectValue placeholder="Select chassis robot" />
              </SelectTrigger>
              <SelectContent>
                {mobileRobots.length === 0 ? (
                  <SelectItem value="_none" disabled>
                    No chassis detected
                  </SelectItem>
                ) : (
                  mobileRobots.map((robot) => {
                    const idx = connectedRobots.indexOf(robot);
                    return (
                      <SelectItem
                        key={`chassis-${idx}`}
                        value={idx.toString()}
                      >
                        {robot.name} ({robot.device_name})
                      </SelectItem>
                    );
                  })
                )}
              </SelectContent>
            </Select>
          </div>
          <div className="space-y-2 w-[180px]">
            <Label>Speed: {(chassisSpeed * 100).toFixed(0)}%</Label>
            <Slider
              value={[chassisSpeed]}
              onValueChange={(v) => setChassisSpeed(v[0])}
              min={0.1}
              max={1.0}
              step={0.1}
            />
          </div>
        </div>

        <div className="flex flex-col md:flex-row gap-4">
          <Button
            onClick={handleChassisStart}
            disabled={selectedChassisId === null || isChassisActive}
            variant={isChassisActive ? "outline" : "default"}
          >
            {!isChassisActive && <Play className="mr-2 h-4 w-4" />}
            {isChassisActive
              ? "Chassis control active"
              : "Start chassis control"}
          </Button>
          <Button
            onClick={handleChassisStop}
            disabled={!isChassisActive}
            variant="destructive"
          >
            <Square className="mr-2 h-4 w-4" />
            Stop chassis
          </Button>
        </div>

        {isChassisActive && (
          <div className="rounded-lg border bg-muted/40 p-4">
            <div className="grid grid-cols-3 gap-2 w-fit mx-auto">
              <div />
              <div
                className={`flex h-12 w-12 items-center justify-center rounded-md border-2 text-lg font-bold select-none transition-colors ${
                  ["ArrowUp", "w", "W"].includes(activeChassisKey ?? "")
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-background"
                }`}
              >
                ↑
              </div>
              <div />
              <div
                className={`flex h-12 w-12 items-center justify-center rounded-md border-2 text-lg font-bold select-none transition-colors ${
                  ["ArrowLeft", "a", "A"].includes(activeChassisKey ?? "")
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-background"
                }`}
              >
                ←
              </div>
              <div className="flex h-12 w-12 items-center justify-center rounded-md border bg-muted text-muted-foreground">
                ●
              </div>
              <div
                className={`flex h-12 w-12 items-center justify-center rounded-md border-2 text-lg font-bold select-none transition-colors ${
                  ["ArrowRight", "d", "D"].includes(activeChassisKey ?? "")
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-background"
                }`}
              >
                →
              </div>
              <div />
              <div
                className={`flex h-12 w-12 items-center justify-center rounded-md border-2 text-lg font-bold select-none transition-colors ${
                  ["ArrowDown", "s", "S"].includes(activeChassisKey ?? "")
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-background"
                }`}
              >
                ↓
              </div>
              <div />
              <div
                className={`flex h-12 w-12 items-center justify-center rounded-md border-2 text-sm font-bold select-none transition-colors ${
                  activeChassisKey === "q" || activeChassisKey === "Q"
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-background"
                }`}
              >
                Q ↺
              </div>
              <div className="flex h-12 w-12 items-center justify-center rounded-md border bg-muted text-muted-foreground text-xs">
                Rotate
              </div>
              <div
                className={`flex h-12 w-12 items-center justify-center rounded-md border-2 text-sm font-bold select-none transition-colors ${
                  activeChassisKey === "e" || activeChassisKey === "E"
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border bg-background"
                }`}
              >
                E ↻
              </div>
            </div>
            <p className="mt-3 text-center text-sm text-muted-foreground">
              ↑/W Fwd | ↓/S Back | ←/A Left | →/D Right | Q ↺ E Rotate
            </p>
          </div>
        )}

        {mobileRobots.length === 0 && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>No chassis detected</AlertTitle>
            <AlertDescription>
              Ensure LeKiwi chassis is connected (IP + port) and add robot in
              admin/connection page.
            </AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  );
}
