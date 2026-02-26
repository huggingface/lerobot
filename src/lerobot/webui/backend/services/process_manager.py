"""Process management service for handling subprocess lifecycle."""

import asyncio
import signal
import uuid
from collections import deque
from datetime import datetime
from typing import AsyncGenerator, Dict, Optional

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

        # Create subprocess with pipes for stdout/stderr
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # Merge stderr into stdout
            env=env,
        )

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
                process_info.logs.append(log_line)

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
                # Already stopped
                process_info.stopped_at = datetime.now()
                return True

        # Send SIGTERM
        try:
            process_info.process.send_signal(signal.SIGTERM)
        except ProcessLookupError:
            # Process already died
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

            # Determine state
            if process_info.process.returncode is not None:
                state = ProcessState.ERROR if process_info.process.returncode != 0 else ProcessState.STOPPED
            else:
                state = ProcessState.RUNNING

            # Calculate uptime
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

        # First, yield existing logs
        async with self._lock:
            for log in process_info.logs:
                yield log

        # Then stream new logs
        last_count = len(process_info.logs)

        while True:
            await asyncio.sleep(0.1)  # Poll every 100ms

            current_count = len(process_info.logs)

            if current_count > last_count:
                async with self._lock:
                    new_logs = list(process_info.logs)[last_count:]

                for log in new_logs:
                    yield log

                last_count = current_count

            # Stop if process has ended and no more logs
            if process_info.process.returncode is not None and current_count == last_count:
                break

    async def get_active_processes(self) -> Dict[str, ProcessStatus]:
        """Get all active processes.

        Returns:
            Dictionary mapping process_id to ProcessStatus.
        """
        active = {}

        async with self._lock:
            for process_id in list(self.processes.keys()):
                status = await self.get_status(process_id)
                if status and status.state == ProcessState.RUNNING:
                    active[process_id] = status

        return active

    async def cleanup(self) -> None:
        """Stop all running processes and cleanup."""
        async with self._lock:
            for process_id in list(self.processes.keys()):
                await self.stop_process(process_id)


# Global process manager instance
process_manager = ProcessManager()
