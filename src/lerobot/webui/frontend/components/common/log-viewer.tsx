"use client";

import { useEffect, useRef } from "react";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Button } from "@/components/ui/button";
import { Download, Trash2 } from "lucide-react";

interface LogViewerProps {
  logs: string[];
  isConnected: boolean;
  onClear?: () => void;
  maxHeight?: string;
}

export function LogViewer({
  logs,
  isConnected,
  onClear,
  maxHeight = "400px",
}: LogViewerProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  const handleDownload = () => {
    const blob = new Blob([logs.join("\n")], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `lerobot-logs-${Date.now()}.txt`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="rounded-md border bg-zinc-950">
      <div className="flex items-center justify-between border-b px-3 py-2">
        <div className="flex items-center gap-2">
          <div
            className={`h-2 w-2 rounded-full ${
              isConnected ? "bg-green-500" : "bg-zinc-500"
            }`}
          />
          <span className="text-xs text-zinc-400">
            {isConnected ? "Connected" : "Disconnected"}
          </span>
          <span className="text-xs text-zinc-600">
            {logs.length} lines
          </span>
        </div>
        <div className="flex gap-1">
          {onClear && (
            <Button
              variant="ghost"
              size="icon"
              className="h-6 w-6 text-zinc-400 hover:text-zinc-200"
              onClick={onClear}
            >
              <Trash2 className="h-3 w-3" />
            </Button>
          )}
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6 text-zinc-400 hover:text-zinc-200"
            onClick={handleDownload}
            disabled={logs.length === 0}
          >
            <Download className="h-3 w-3" />
          </Button>
        </div>
      </div>
      <ScrollArea style={{ height: maxHeight }}>
        <div className="p-3 font-mono text-xs leading-5">
          {logs.length === 0 ? (
            <span className="text-zinc-600">Waiting for output...</span>
          ) : (
            logs.map((line, i) => (
              <div
                key={i}
                className={`whitespace-pre-wrap break-all ${
                  line.toLowerCase().includes("error")
                    ? "text-red-400"
                    : line.toLowerCase().includes("warning")
                      ? "text-yellow-400"
                      : "text-zinc-300"
                }`}
              >
                {line}
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>
      </ScrollArea>
    </div>
  );
}
