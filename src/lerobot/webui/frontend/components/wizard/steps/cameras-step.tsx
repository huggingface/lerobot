"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import { Loader2, Camera } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { CAMERA_NAME_OPTIONS } from "@/lib/wizard-types";
import { useWizard } from "../wizard-provider";
import { StepCard } from "../step-card";

/** Keywords that identify built-in / phone cameras to exclude */
const BUILTIN_KEYWORDS = [
  "facetime",
  "built-in",
  "macbook",
  "iphone",
  "ipad",
  "continuity",
  "ir camera",
  "infrared",
];

function isExternalCamera(label: string): boolean {
  const lower = label.toLowerCase();
  return !BUILTIN_KEYWORDS.some((kw) => lower.includes(kw));
}

/** Live video feed for a single camera using getUserMedia */
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
    <div className="border-t bg-muted/30 p-2">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
        className="w-full rounded"
      />
    </div>
  );
}

export function CamerasStep() {
  const { state, dispatch } = useWizard();
  const [detecting, setDetecting] = useState(false);

  const selectedCameras = state.cameraSelections.filter((c) => c.included);
  const allNamed =
    selectedCameras.length > 0 && selectedCameras.every((c) => c.name !== "");

  // Names already used by other cameras
  const getUsedNames = useCallback(
    (excludeDeviceId: string): Set<string> => {
      const used = new Set<string>();
      for (const cam of state.cameraSelections) {
        if (
          cam.deviceId !== excludeDeviceId &&
          cam.included &&
          cam.name
        ) {
          used.add(cam.name);
        }
      }
      return used;
    },
    [state.cameraSelections]
  );

  async function detectCameras() {
    setDetecting(true);
    try {
      // Request permission first (labels are empty without permission)
      const tempStream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });
      tempStream.getTracks().forEach((t) => t.stop());

      const devices = await navigator.mediaDevices.enumerateDevices();
      const cameras = devices
        .filter((d) => d.kind === "videoinput")
        .filter((d) => isExternalCamera(d.label))
        .map((d) => ({
          deviceId: d.deviceId,
          label: d.label || `Camera ${d.deviceId.slice(0, 8)}`,
        }));

      dispatch({ type: "SET_DETECTED_CAMERAS", cameras });
    } catch (err) {
      console.error("Failed to detect cameras", err);
    } finally {
      setDetecting(false);
    }
  }

  return (
    <StepCard
      title="Select Cameras"
      description="Detect cameras and assign a name to each one you want to use. Toggle a camera on to see its live feed."
      nextDisabled={!allNamed}
    >
      <div className="space-y-6">
        <Button
          variant="outline"
          onClick={detectCameras}
          disabled={detecting}
          className="w-full"
        >
          {detecting ? (
            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
          ) : (
            <Camera className="mr-2 h-4 w-4" />
          )}
          {detecting ? "Detecting..." : "Detect Cameras"}
        </Button>

        {state.cameraSelections.length > 0 && (
          <div className="space-y-4">
            <p className="text-sm text-muted-foreground">
              {state.cameraSelections.length} camera(s) found. Select which to
              include:
            </p>
            {state.cameraSelections.map((cam) => {
              const usedNames = getUsedNames(cam.deviceId);
              return (
                <div
                  key={cam.deviceId}
                  className="overflow-hidden rounded-lg border"
                >
                  <div className="flex items-center gap-4 p-4">
                    <Switch
                      checked={cam.included}
                      onCheckedChange={(checked) =>
                        dispatch({
                          type: "TOGGLE_CAMERA",
                          deviceId: cam.deviceId,
                          included: checked,
                        })
                      }
                    />
                    <div className="flex-1">
                      <p className="text-sm font-medium">{cam.label}</p>
                    </div>
                    {cam.included && (
                      <div className="w-40">
                        <Select
                          value={cam.name || ""}
                          onValueChange={(name) =>
                            dispatch({
                              type: "SET_CAMERA_NAME",
                              deviceId: cam.deviceId,
                              name,
                            })
                          }
                        >
                          <SelectTrigger>
                            <SelectValue placeholder="Name..." />
                          </SelectTrigger>
                          <SelectContent>
                            {CAMERA_NAME_OPTIONS.filter(
                              (n) => !usedNames.has(n)
                            ).map((name) => (
                              <SelectItem key={name} value={name}>
                                {name}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      </div>
                    )}
                  </div>
                  {cam.included && <CameraFeed deviceId={cam.deviceId} />}
                </div>
              );
            })}
          </div>
        )}

        {state.cameraSelections.length === 0 && !detecting && (
          <p className="text-center text-sm text-muted-foreground">
            Click &quot;Detect Cameras&quot; to find connected cameras.
          </p>
        )}
      </div>
    </StepCard>
  );
}
