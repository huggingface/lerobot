import json
import logging
import socket
import time
import threading
import queue
import struct
import select
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
        log_file: str = "udp_sender.log",
        enable_logging: bool = False,
    ):
        """Args
        ----
        server: str
            Remote endpoint in the form "<ip>:<port>" (e.g. "10.10.10.10:5555").
        log_file: str
            Path to log file for tracking send timestamps.
        enable_logging: bool
            Whether to enable logging at all. Set to False for minimum latency.
        """
        ip, port_str = server.split(":")
        self._addr: Tuple[str, int] = (ip, int(port_str))
        self._enable_logging = enable_logging

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Non-blocking send – we never wait for ACKs.
        self._sock.setblocking(False)

        # Increase send buffer size to prevent blocking
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 65536)

        # Packet sequence number for identification
        self._sequence_number = 0

        # Setup logging only if enabled
        if self._enable_logging:
            self._logger = logging.getLogger(f"UDPSender_{ip}_{port_str}")
            self._logger.setLevel(logging.INFO)

            # Create file handler if not already exists
            if not self._logger.handlers:
                handler = AsyncLogHandler(log_file)
                self._logger.addHandler(handler)
                self._logger.propagate = False
        else:
            self._logger = None

    def send(self, action: Dict[str, float]) -> None:
        """Serialise *action* to binary format and transmit it over UDP."""
        send_timestamp = time.time()

        # Binary format for maximum performance
        payload = self._serialize_binary(action, send_timestamp)

        # Best-effort – if the socket is not ready we'll simply drop the frame.
        try:
            self._sock.sendto(payload, self._addr)
            if self._enable_logging and self._logger:
                self._logger.info(
                    f"SENT - packet_id: {self._sequence_number}, timestamp: {send_timestamp:.6f}, payload_size: {len(payload)} bytes, action: {action}"
                )
        except (BlockingIOError, OSError) as e:
            # Dropped – nothing to do, next frame will go through.
            if self._enable_logging and self._logger:
                self._logger.warning(
                    f"DROPPED - packet_id: {self._sequence_number}, timestamp: {send_timestamp:.6f}, error: {e}, payload_size: {len(payload)} bytes, action: {action}"
                )

        # Increment sequence number for next packet
        self._sequence_number += 1

    def _serialize_binary(self, action: Dict[str, float], timestamp: float) -> bytes:
        """Serialize action to binary format for maximum performance."""
        # Pre-calculate total size to avoid memory reallocations
        total_size = 20  # header: magic(4) + seq(4) + timestamp(8) + count(4)
        key_sizes = []
        for key in action:
            key_bytes = key.encode("utf-8")
            key_sizes.append((key, key_bytes))
            total_size += 4 + len(key_bytes) + 8  # key_len(4) + key + value(8)

        # Allocate buffer once
        payload = bytearray(total_size)
        offset = 0

        # Write header
        payload[offset : offset + 4] = b"UDPA"  # magic
        offset += 4
        struct.pack_into("<I", payload, offset, self._sequence_number)
        offset += 4
        struct.pack_into("<d", payload, offset, timestamp)
        offset += 8
        struct.pack_into("<I", payload, offset, len(action))
        offset += 4

        # Write action data
        for key, key_bytes in key_sizes:
            struct.pack_into("<I", payload, offset, len(key_bytes))
            offset += 4
            payload[offset : offset + len(key_bytes)] = key_bytes
            offset += len(key_bytes)
            struct.pack_into("<d", payload, offset, action[key])
            offset += 8

        return bytes(payload)


class UDPTransportReceiver:
    """Blocking UDP receiver used by the teleoperation *follower* machine."""

    def __init__(
        self,
        port: int,
        buffer_size: int = 65535,
        log_file: str = "udp_receiver.log",
        enable_logging: bool = False,
        drop_old_packets: bool = True,
        latency_monitor_interval: int = 100,
    ):
        """Listen on *port* for incoming action packets.

        Args
        ----
        port: int
            Port to listen on for incoming packets.
        buffer_size: int
            Maximum size of UDP packets to receive.
        log_file: str
            Path to log file for tracking receive timestamps.
        enable_logging: bool
            Whether to enable logging at all. Set to False for minimum latency.
        drop_old_packets: bool
            Whether to drop old packets when buffer is full to maintain low latency.
        latency_monitor_interval: int
            Print latency every N packets (0 to disable). Use 10-50 for monitoring.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow immediate rebinding after a restart.
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

        # Increase receive buffer size significantly
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)  # 256KB
        
        # Set socket to non-blocking mode for faster processing
        self._sock.setblocking(False)
        
        # Optimize for low latency
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_PRIORITY, 6)  # High priority
        
        # Bind on all interfaces – users can rely on firewall rules for protection.
        self._sock.bind(("", port))
        self._buffer_size = buffer_size
        self._enable_logging = enable_logging
        self._drop_old_packets = drop_old_packets
        self._latency_monitor_interval = latency_monitor_interval

        # Track last received packet ID to detect gaps
        self._last_packet_id = -1
        self._packet_count = 0

        # Setup logging only if enabled
        if self._enable_logging:
            self._logger = logging.getLogger(f"UDPReceiver_{port}")
            self._logger.setLevel(logging.INFO)

            # Create file handler if not already exists
            if not self._logger.handlers:
                handler = AsyncLogHandler(log_file)
                self._logger.addHandler(handler)
                self._logger.propagate = False
        else:
            self._logger = None

    def recv(self) -> Dict[str, float]:
        """Block until the next action packet is received and return it."""
        # Use select with very short timeout for non-blocking mode
        ready = select.select([self._sock], [], [], 0.001)  # 1ms timeout
        if not ready[0]:
            return {}  # No data available

        payload, addr = self._sock.recvfrom(self._buffer_size)
        recv_timestamp = time.time()

        try:
            action, packet_id, send_timestamp = self._deserialize_binary(payload)
            latency_ms = (recv_timestamp - send_timestamp) * 1000

            # Simple latency monitoring - print every 100th packet (less frequent)
            if packet_id is not None and packet_id % 100 == 0:
                print(f"{latency_ms:.1f}")  # Simple format

            return action

        except (struct.error, UnicodeDecodeError):
            # Just return empty action on error, don't log
            return {}

    def _deserialize_binary(
        self, payload: bytes
    ) -> Tuple[Dict[str, float], int, float]:
        """Deserialize binary payload for maximum performance."""
        # Format: [magic:4][seq:4][timestamp:8][action_count:4][key1_len:4][key1:str][val1:8][key2_len:4][key2:str][val2:8]...
        if len(payload) < 20:  # Minimum size for header
            raise struct.error("Payload too short")

        magic = payload[:4]
        if magic != b"UDPA":
            raise struct.error("Invalid magic number")

        packet_id = struct.unpack("<I", payload[4:8])[0]
        send_timestamp = struct.unpack("<d", payload[8:16])[0]
        action_count = struct.unpack("<I", payload[16:20])[0]

        # Pre-allocate action dict with expected size
        action = {}
        offset = 20

        for _ in range(action_count):
            if offset + 4 > len(payload):
                raise struct.error("Payload truncated")

            key_len = struct.unpack("<I", payload[offset : offset + 4])[0]
            offset += 4

            if offset + key_len + 8 > len(payload):
                raise struct.error("Payload truncated")

            # Decode key once
            key = payload[offset : offset + key_len].decode("utf-8")
            offset += key_len

            # Get value
            value = struct.unpack("<d", payload[offset : offset + 8])[0]
            offset += 8

            action[key] = value

        return action, packet_id, send_timestamp
