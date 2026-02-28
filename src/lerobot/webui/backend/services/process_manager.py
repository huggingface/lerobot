"""Process management service for handling subprocess lifecycle."""

import asyncio
import os
import re
import signal
import uuid
from collections import deque
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional

# Regex to strip ANSI escape sequences from subprocess output
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

from lerobot.webui.backend.models.system import ProcessState, ProcessStatus


class ProcessInfo:
    """Internal process information."""

    def __init__(self, process: asyncio.subprocess.Process, process_type: str):
        """Initialize ProcessInfo.

        Args:
            process: Asyncio subprocess.
            process_type: Type of process (teleoperation, recording, calibration).
        """
        self.process = process
        self.process_type = process_type
        self.started_at = datetime.now()
        self.stopped_at: Optional[datetime] = None
        self.logs: deque = deque(maxlen=1000)  # Keep last 1000 log lines
        self.log_seq: int = 0  # Monotonic counter incremented on each append
        self.log_event: asyncio.Event = asyncio.Event()  # Signalled on new log lines
        self.error_message: Optional[str] = None
        self.log_task: Optional[asyncio.Task] = None


class ProcessManager:
    """Manages subprocess lifecycle for CLI commands."""

    def __init__(self):
        """Initialize ProcessManager."""
        self.processes: Dict[str, ProcessInfo] = {}
        self._lock = asyncio.Lock()

    async def start_process(
        self, command: list[str], process_type: str, env: Optional[Dict[str, str]] = None
    ) -> str:
        """Start a subprocess.

        Args:
            command: Command to execute as list of strings.
            process_type: Process type (teleoperation, recording, calibration).
            env: Optional environment variables.

        Returns:
            Process ID for tracking.
        """
        process_id = str(uuid.uuid4())

        # Build environment: inherit parent env, force unbuffered Python output
        # so that subprocess print() calls flush immediately to the pipe
        # (otherwise Python uses block buffering when stdout is not a TTY).
        proc_env = os.environ.copy()
        proc_env["PYTHONUNBUFFERED"] = "1"
        if env:
            proc_env.update(env)

        # Create subprocess with pipes for stdout/stderr.
        # Pipe stdin so we can feed newlines — this auto-accepts calibration
        # prompts like "Press ENTER to use provided calibration file".
        process = await asyncio.create_subprocess_exec(
            *command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
            env=proc_env,
        )

        # Send a few newlines to satisfy any interactive prompts, then close stdin
        if process.stdin:
            process.stdin.write(b"\n\n\n")
            await process.stdin.drain()
            process.stdin.close()

        process_info = ProcessInfo(process, process_type)

        async with self._lock:
            self.processes[process_id] = process_info

        # Start log collection task
        process_info.log_task = asyncio.create_task(
            self._collect_logs(process_id, process_info)
        )

        return process_id

    async def _collect_logs(self, process_id: str, process_info: ProcessInfo) -> None:
        """Collect logs from subprocess stdout/stderr.

        Args:
            process_id: Process identifier.
            process_info: Process information object.
        """
        try:
            while True:
                line = await process_info.process.stdout.readline()
                if not line:
                    break

                log_line = line.decode("utf-8", errors="replace").rstrip()
                # Strip ANSI escape sequences (e.g. cursor-up codes from teleoperation)
                log_line = _ANSI_ESCAPE_RE.sub("", log_line)
                if not log_line:
                    continue
                process_info.logs.append(log_line)
                process_info.log_seq += 1
                # Wake up any WebSocket stream_logs waiters
                process_info.log_event.set()
                process_info.log_event.clear()

        except asyncio.CancelledError:
            pass
        except Exception as e:
            process_info.error_message = f"Error collecting logs: {e}"

    async def stop_process(self, process_id: str, timeout: float = 5.0) -> bool:
        """Stop a running subprocess.

        Args:
            process_id: Process identifier.
            timeout: Timeout in seconds before force kill.

        Returns:
            True if stopped successfully, False otherwise.
        """
        async with self._lock:
            process_info = self.processes.get(process_id)

            if not process_info:
                return False

            if process_info.process.returncode is not None:
                process_info.stopped_at = datetime.now()
                return True

        return await self._stop_process_unlocked(process_info)

    async def _stop_process_unlocked(self, process_info: ProcessInfo, timeout: float = 5.0) -> bool:
        """Stop a process without acquiring the lock. Caller must ensure safe access."""
        if process_info.process.returncode is not None:
            process_info.stopped_at = datetime.now()
            return True

        # Send SIGTERM
        try:
            process_info.process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            process_info.stopped_at = datetime.now()
            return True

        # Wait for graceful shutdown
        try:
            await asyncio.wait_for(process_info.process.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            # Force kill with SIGKILL
            try:
                process_info.process.kill()
                await process_info.process.wait()
            except ProcessLookupError:
                pass

        process_info.stopped_at = datetime.now()

        # Cancel log collection task
        if process_info.log_task and not process_info.log_task.done():
            process_info.log_task.cancel()

        return True

    async def get_status(self, process_id: str) -> Optional[ProcessStatus]:
        """Get process status.

        Args:
            process_id: Process identifier.

        Returns:
            ProcessStatus or None if not found.
        """
        async with self._lock:
            process_info = self.processes.get(process_id)
            if not process_info:
                return None
            return self._build_status(process_id, process_info)

    def _build_status(self, process_id: str, process_info: ProcessInfo) -> ProcessStatus:
        """Build a ProcessStatus from a ProcessInfo. Does not acquire the lock."""
        if process_info.process.returncode is not None:
            state = ProcessState.ERROR if process_info.process.returncode != 0 else ProcessState.STOPPED
        else:
            state = ProcessState.RUNNING

        uptime_seconds = None
        if state == ProcessState.RUNNING:
            uptime_seconds = (datetime.now() - process_info.started_at).total_seconds()

        return ProcessStatus(
            process_id=process_id,
            process_type=process_info.process_type,
            state=state,
            pid=process_info.process.pid,
            started_at=process_info.started_at,
            stopped_at=process_info.stopped_at,
            uptime_seconds=uptime_seconds,
            error_message=process_info.error_message,
        )

    async def get_logs(self, process_id: str, last_n: Optional[int] = None) -> list[str]:
        """Get process logs.

        Args:
            process_id: Process identifier.
            last_n: Number of last log lines to return. None returns all.

        Returns:
            List of log lines.
        """
        async with self._lock:
            process_info = self.processes.get(process_id)

            if not process_info:
                return []

            logs = list(process_info.logs)

            if last_n is not None:
                logs = logs[-last_n:]

            return logs

    async def stream_logs(self, process_id: str) -> AsyncGenerator[str, None]:
        """Stream logs from process in real-time.

        Args:
            process_id: Process identifier.

        Yields:
            Log lines as they arrive.
        """
        process_info = self.processes.get(process_id)

        if not process_info:
            return

        # First, yield existing logs and record current sequence number
        async with self._lock:
            for log in process_info.logs:
                yield log
            last_seq = process_info.log_seq

        # Then stream new logs using the monotonic sequence counter
        # (len()-based tracking breaks when the deque is full and evicting old entries)
        while True:
            # Wait for new log lines or timeout to check process exit
            try:
                await asyncio.wait_for(process_info.log_event.wait(), timeout=0.5)
            except asyncio.TimeoutError:
                pass

            current_seq = process_info.log_seq
            if current_seq > last_seq:
                # Number of new lines added since we last checked
                new_count = current_seq - last_seq
                # Grab the tail of the deque (new_count items, clamped to deque size)
                async with self._lock:
                    all_logs = list(process_info.logs)
                new_logs = all_logs[-min(new_count, len(all_logs)):]

                for log in new_logs:
                    yield log

                last_seq = current_seq

            # Stop if process has ended and no more logs
            if process_info.process.returncode is not None and current_seq == process_info.log_seq:
                break

    async def get_active_processes(self) -> Dict[str, ProcessStatus]:
        """Get all active processes.

        Returns:
            Dictionary mapping process_id to ProcessStatus.
        """
        active = {}

        async with self._lock:
            for process_id, process_info in self.processes.items():
                status = self._build_status(process_id, process_info)
                if status.state == ProcessState.RUNNING:
                    active[process_id] = status

        return active

    async def cleanup(self) -> None:
        """Stop all running processes and cleanup."""
        async with self._lock:
            processes_to_stop = list(self.processes.values())

        for process_info in processes_to_stop:
            await self._stop_process_unlocked(process_info)


# Global process manager instance
process_manager = ProcessManager()
