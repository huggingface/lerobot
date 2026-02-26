"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const WS_BASE = process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000";

export function useWebSocket(processId: string | null) {
  const [logs, setLogs] = useState<string[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!processId) {
      setIsConnected(false);
      return;
    }

    const ws = new WebSocket(`${WS_BASE}/ws/logs/${processId}`);
    wsRef.current = ws;

    ws.onopen = () => setIsConnected(true);

    ws.onmessage = (event) => {
      setLogs((prev) => [...prev.slice(-999), event.data]);
    };

    ws.onclose = () => setIsConnected(false);

    ws.onerror = () => setIsConnected(false);

    return () => {
      ws.close();
      wsRef.current = null;
    };
  }, [processId]);

  const clearLogs = useCallback(() => setLogs([]), []);

  return { logs, isConnected, clearLogs };
}
