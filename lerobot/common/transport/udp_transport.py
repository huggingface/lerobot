import json
import logging
import socket
import time
import threading
import queue
import struct
import select
import os
from collections import deque
from typing import Dict, Tuple, Optional

__all__ = [
    "UDPTransportSender",
    "UDPTransportReceiver",
]

class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler to prevent file I/O from blocking the main thread."""

    def __init__(self, log_file: str, max_queue_size: int = 1000):
        super().__init__()
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s.%(msecs)03d - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._handler.setFormatter(formatter)

        # Start background thread
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def _worker(self):
        """Background thread that handles actual file writing."""
        while True:
            try:
                record = self._queue.get()
                if record is None:  # Shutdown signal
                    break
                self._handler.emit(record)
            except Exception:
                pass  # Don't let logging errors crash the worker

    def emit(self, record):
        """Queue a log record for async processing."""
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            # Drop oldest log if queue is full
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(record)
            except queue.Empty:
                pass

    def close(self):
        """Shutdown the async handler."""
        self._queue.put(None)
        self._thread.join(timeout=1.0)
        self._handler.close()
        super().close()


class UDPTransportSender:
    """Lightweight, fire-and-forget UDP sender used by the teleoperation *leader* machine."""

    def __init__(
        self,
        server: str,
        enable_logging: bool = False,
        enable_diagnostics: bool = False,
        log_to_file: bool = True,
        log_to_stdout: bool = False,
        log_file: str = "udp_sender.log",
        log_interval: int = 50,
    ):
        """Initialize UDP sender.
        
        Args:
            server: Remote endpoint "<ip>:<port>" (e.g. "10.10.10.10:5555")
            enable_logging: Enable detailed packet logging
            enable_diagnostics: Enable diagnostic output every N packets
            log_to_file: Write logs to file
            log_to_stdout: Write logs to stdout
            log_file: Path to log file
            log_interval: Log every N packets (0 = log all packets)
        """
        ip, port_str = server.split(":")
        self._addr: Tuple[str, int] = (ip, int(port_str))
        self._enable_logging = enable_logging
        self._enable_diagnostics = enable_diagnostics
        self._log_to_file = log_to_file
        self._log_to_stdout = log_to_stdout
        self._log_interval = log_interval

        # Setup socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setblocking(False)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

        # Packet tracking
        self._sequence_number = 0
        self._last_diag_time = time.time()

        # Setup logging
        self._logger = None
        if self._enable_logging and self._log_to_file:
            self._logger = logging.getLogger(f"UDPSender_{ip}_{port_str}")
            self._logger.setLevel(logging.INFO)
            if not self._logger.handlers:
                handler = AsyncLogHandler(log_file)
                self._logger.addHandler(handler)
                self._logger.propagate = False

    def send(self, action: Dict[str, float]) -> None:
        """Send action over UDP."""
        send_timestamp = time.time()
        payload = self._serialize_binary(action, send_timestamp)

        try:
            self._sock.sendto(payload, self._addr)
            
            # Diagnostics every 50th packet
            if self._enable_diagnostics and self._sequence_number % 50 == 0:
                self._log_diagnostics(send_timestamp, len(payload))
                
            # Detailed logging (configurable interval)
            if self._enable_logging:
                if self._log_interval == 0 or self._sequence_number % self._log_interval == 0:
                    self._log_packet("SENT", send_timestamp, len(payload), action)
                
        except (BlockingIOError, OSError) as e:
            if self._enable_logging:
                self._log_packet("DROPPED", send_timestamp, len(payload), action, error=str(e))

        self._sequence_number += 1

    def _log_diagnostics(self, send_timestamp: float, payload_size: int) -> None:
        """Log diagnostic information."""
        current_time = time.time()
        time_since_last = (current_time - self._last_diag_time) * 1000
        
        msg = f"SENDER: packet={self._sequence_number}, time_since_last={time_since_last:.1f}ms, payload_size={payload_size}B"
        
        if self._log_to_stdout:
            print(msg)
        if self._logger:
            self._logger.info(msg)
            
        self._last_diag_time = current_time

    def _log_packet(self, status: str, timestamp: float, payload_size: int, action: Dict[str, float], error: str = None) -> None:
        """Log individual packet information."""
        if error:
            msg = f"{status} - packet_id: {self._sequence_number}, timestamp: {timestamp:.6f}, error: {error}, payload_size: {payload_size} bytes"
        else:
            msg = f"{status} - packet_id: {self._sequence_number}, timestamp: {timestamp:.6f}, payload_size: {payload_size} bytes, action: {action}"
        
        if self._log_to_stdout:
            print(msg)
        if self._logger:
            if error:
                self._logger.warning(msg)
            else:
                self._logger.info(msg)

    def _serialize_binary(self, action: Dict[str, float], timestamp: float) -> bytes:
        """Serialize action to binary format."""
        # Calculate total size
        total_size = 20  # header: magic(4) + seq(4) + timestamp(8) + count(4)
        key_data = []
        for key in action:
            key_bytes = key.encode("utf-8")
            key_data.append((key, key_bytes))
            total_size += 4 + len(key_bytes) + 8  # key_len(4) + key + value(8)

        # Allocate buffer
        payload = bytearray(total_size)
        offset = 0

        # Write header
        payload[offset:offset + 4] = b"UDPA"
        offset += 4
        struct.pack_into("<I", payload, offset, self._sequence_number)
        offset += 4
        struct.pack_into("<d", payload, offset, timestamp)
        offset += 8
        struct.pack_into("<I", payload, offset, len(action))
        offset += 4

        # Write action data
        for key, key_bytes in key_data:
            struct.pack_into("<I", payload, offset, len(key_bytes))
            offset += 4
            payload[offset:offset + len(key_bytes)] = key_bytes
            offset += len(key_bytes)
            struct.pack_into("<d", payload, offset, action[key])
            offset += 8

        return bytes(payload)


class UDPTransportReceiver:
    """UDP receiver used by the teleoperation *follower* machine."""

    def __init__(
        self,
        port: int,
        buffer_size: int = 65535,
        enable_logging: bool = True,
        enable_diagnostics: bool = False,
        log_to_file: bool = False,
        log_to_stdout: bool = True,
        log_file: str = "udp_receiver.log",
        drop_old_packets: bool = True,
        log_interval: int = 50,
    ):
        """Initialize UDP receiver.
        
        Args:
            port: Port to listen on for incoming packets
            buffer_size: Maximum size of UDP packets to receive
            enable_logging: Enable detailed packet logging
            enable_diagnostics: Enable diagnostic output every N packets
            log_to_file: Write logs to file
            log_to_stdout: Write logs to stdout
            log_file: Path to log file
            drop_old_packets: Whether to drop old packets to maintain low latency
            log_interval: Log every N packets (0 = log all packets)
        """
        # Setup socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)  # 256KB
        self._sock.setblocking(False)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 6)  # High priority
        self._sock.bind(("", port))
        
        # Configuration
        self._buffer_size = buffer_size
        self._enable_logging = enable_logging
        self._enable_diagnostics = enable_diagnostics
        self._log_to_file = log_to_file
        self._log_to_stdout = log_to_stdout
        self._drop_old_packets = drop_old_packets
        self._log_interval = log_interval

        # Packet tracking
        self._last_packet_id = -1
        self._packet_count = 0
        
        # Diagnostic counters
        self._last_diag_time = time.time()
        self._packets_since_diag = 0

        # Setup logging
        self._logger = None
        if self._enable_logging and self._log_to_file:
            self._logger = logging.getLogger(f"UDPReceiver_{port}")
            self._logger.setLevel(logging.INFO)
            if not self._logger.handlers:
                handler = AsyncLogHandler(log_file)
                self._logger.addHandler(handler)
                self._logger.propagate = False

    def recv(self) -> Dict[str, float]:
        """Receive and return the freshest action packet."""
        latest_action = None
        latest_packet_id = None
        latest_latency = float('inf')
        packets_processed = 0
        
        # Process all available packets
        while True:
            ready = select.select([self._sock], [], [], 0.0)  # Non-blocking
            if not ready[0]:
                break

            payload, addr = self._sock.recvfrom(self._buffer_size)
            recv_timestamp = time.time()
            packets_processed += 1

            # Fast-path: echo latency-probe packets
            if payload.startswith(PING_MAGIC):
                self._sock.sendto(payload, addr)
                continue

            try:
                action, packet_id, send_timestamp = self._deserialize_binary(payload)
                latency_ms = (recv_timestamp - send_timestamp) * 1000

                # Keep the packet with lowest latency (freshest)
                if latency_ms < latest_latency:
                    latest_action = action
                    latest_packet_id = packet_id
                    latest_latency = latency_ms

            except (struct.error, UnicodeDecodeError):
                if self._enable_logging:
                    self._log_error("DECODE_ERROR", recv_timestamp, addr, payload)
                continue

        # Process the freshest packet
        if latest_action is not None:
            # Diagnostics every 50th packet
            if self._enable_diagnostics and latest_packet_id % 50 == 0:
                self._log_diagnostics(latest_packet_id, latest_latency, packets_processed)
                
            # Detailed logging (configurable interval)
            if self._enable_logging:
                if self._log_interval == 0 or latest_packet_id % self._log_interval == 0:
                    self._log_packet("RECEIVED", latest_packet_id, recv_timestamp, send_timestamp, 
                                   latest_latency, len(payload), latest_action)
            
            self._packets_since_diag += 1
            return latest_action
        
        return {}

    def _log_diagnostics(self, packet_id: int, latency: float, packets_processed: int) -> None:
        """Log diagnostic information."""
        current_time = time.time()
        elapsed = current_time - self._last_diag_time
        packet_rate = self._packets_since_diag / elapsed if elapsed > 0 else 0
        
        msg = f"RECEIVER: packet={packet_id}, latency={latency:.1f}ms, rate={packet_rate:.1f}Hz, processed={packets_processed}"
        
        if self._log_to_stdout:
            print(msg)
        if self._logger:
            self._logger.info(msg)
            
        self._last_diag_time = current_time
        self._packets_since_diag = 0

    def _log_packet(self, status: str, packet_id: int, recv_time: float, send_time: float, 
                   latency: float, payload_size: int, action: Dict[str, float]) -> None:
        """Log individual packet information."""
        msg = (f"{status} - packet_id: {packet_id}, recv_timestamp: {recv_time:.6f}, "
               f"send_timestamp: {send_time:.6f}, latency: {latency:.2f}ms, "
               f"payload_size: {payload_size} bytes, action: {action}")
        
        if self._log_to_stdout:
            print(msg)
        if self._logger:
            self._logger.info(msg)

    def _log_error(self, status: str, timestamp: float, addr: Tuple[str, int], payload: bytes) -> None:
        """Log error information."""
        msg = f"{status} - timestamp: {timestamp:.6f}, from: {addr[0]}:{addr[1]}, payload: {payload[:100]}..."
        
        if self._log_to_stdout:
            print(msg)
        if self._logger:
            self._logger.error(msg)

    def _deserialize_binary(self, payload: bytes) -> Tuple[Dict[str, float], int, float]:
        """Deserialize binary payload."""
        if len(payload) < 20:
            raise struct.error("Payload too short")

        magic = payload[:4]
        if magic != b"UDPA":
            raise struct.error("Invalid magic number")

        packet_id = struct.unpack("<I", payload[4:8])[0]
        send_timestamp = struct.unpack("<d", payload[8:16])[0]
        action_count = struct.unpack("<I", payload[16:20])[0]

        action = {}
        offset = 20

        for _ in range(action_count):
            if offset + 4 > len(payload):
                raise struct.error("Payload truncated")

            key_len = struct.unpack("<I", payload[offset:offset + 4])[0]
            offset += 4

            if offset + key_len + 8 > len(payload):
                raise struct.error("Payload truncated")

            key = payload[offset:offset + key_len].decode("utf-8")
            offset += key_len

            value = struct.unpack("<d", payload[offset:offset + 8])[0]
            offset += 8

            action[key] = value

        return action, packet_id, send_timestamp
