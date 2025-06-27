import json
import logging
import socket
import time
import threading
import queue
import struct
import select
import os
import psutil
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
        # Track when the method is called (application-level timing)
        method_call_time = time.time()
        send_timestamp = time.time()

        # Binary format for maximum performance
        payload = self._serialize_binary(action, send_timestamp)

        # Best-effort – if the socket is not ready we'll simply drop the frame.
        try:
            self._sock.sendto(payload, self._addr)
            
            # Sender diagnostics every 50th packet
            if self._sequence_number % 50 == 0:
                current_time = time.time()
                time_since_last = current_time - getattr(self, '_last_send_time', current_time)
                app_delay = (send_timestamp - method_call_time) * 1000  # Time from call to send
                print(f"SENDER: packet={self._sequence_number}, send_time={send_timestamp:.6f}, time_since_last={time_since_last*1000:.1f}ms, app_delay={app_delay:.1f}ms")
                self._last_send_time = current_time
                
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
        
        # Diagnostic counters
        self._last_diag_time = time.time()
        self._packets_since_diag = 0
        
        # System monitoring
        self._process = psutil.Process(os.getpid())
        self._last_cpu_time = self._process.cpu_times()
        self._last_memory = self._process.memory_info()
        
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
        # Drain all available packets and keep the freshest one
        latest_action = None
        latest_packet_id = None
        latest_latency = float('inf')
        packets_processed = 0
        
        # Process all available packets
        while True:
            # Immediate non-blocking check - no timeout
            select_start = time.time()
            ready = select.select([self._sock], [], [], 0.0)  # No timeout
            select_time = (time.time() - select_start) * 1000
            if not ready[0]:
                break  # No more data available

            recv_start = time.time()
            payload, addr = self._sock.recvfrom(self._buffer_size)
            recv_time = (time.time() - recv_start) * 1000
            recv_timestamp = time.time()
            packets_processed += 1

            try:
                deserialize_start = time.time()
                action, packet_id, send_timestamp = self._deserialize_binary(payload)
                deserialize_time = (time.time() - deserialize_start) * 1000
                latency_ms = (recv_timestamp - send_timestamp) * 1000

                # Keep the packet with lowest latency (freshest)
                if latency_ms < latest_latency:
                    latest_action = action
                    latest_packet_id = packet_id
                    latest_latency = latency_ms

            except (struct.error, UnicodeDecodeError):
                continue  # Skip malformed packets

        # Return the freshest packet
        if latest_action is not None:
            # Comprehensive diagnostic logging every 50th packet
            if latest_packet_id is not None and latest_packet_id % 50 == 0:
                current_time = time.time()
                elapsed = current_time - self._last_diag_time
                packet_rate = self._packets_since_diag / elapsed if elapsed > 0 else 0
                
                # Calculate time since packet was sent
                time_since_send = (current_time - send_timestamp) * 1000 if 'send_timestamp' in locals() else 0
                
                # System diagnostics
                current_cpu_time = self._process.cpu_times()
                current_memory = self._process.memory_info()
                cpu_delta = current_cpu_time.user - self._last_cpu_time.user
                memory_delta = current_memory.rss - self._last_memory.rss
                
                print(f"DIAG: packet={latest_packet_id}, latency={latest_latency:.1f}ms, rate={packet_rate:.1f}Hz, processed={packets_processed}")
                print(f"  TIMING: select={select_time:.3f}ms, recv={recv_time:.3f}ms, deserialize={deserialize_time:.3f}ms")
                print(f"  NETWORK: time_since_send={time_since_send:.1f}ms, network_latency={latest_latency:.1f}ms")
                print(f"  OVERHEAD: total_processing={select_time+recv_time+deserialize_time:.3f}ms")
                print(f"  SYSTEM: cpu_delta={cpu_delta:.3f}s, memory_delta={memory_delta/1024:.1f}KB, memory_rss={current_memory.rss/1024/1024:.1f}MB")
                
                self._last_diag_time = current_time
                self._packets_since_diag += 1
                self._last_cpu_time = current_cpu_time
                self._last_memory = current_memory
            
            self._packets_since_diag += 1
            return latest_action
        
        # If no valid packet found, return empty action
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

    def ping(self, addr: Tuple[str, int], bufsize: int = 65535) -> float:
        """Measure round-trip latency to a given address."""
        t0 = time.monotonic_ns()
        self._sock.sendto(b"PING", addr)
        data, _ = self._sock.recvfrom(bufsize)          # echoed ping
        rtt_ms = (time.monotonic_ns() - t0) / 1e6
        one_way_ms = rtt_ms / 2
        print(f"RTT={rtt_ms:.2f} ms  ⇒  one-way ≈ {one_way_ms:.2f} ms")
        return one_way_ms

def ping_task():
    while running:
        pkt = build_ping_packet()        # includes sequence + t_send (monotonic_ns)
        sock.sendto(pkt, addr)
        try:
            echo, _ = sock.recvfrom(buf) # non-blocking / short timeout
            rtt = (time.monotonic_ns() - parse_t_send(echo)) / 2e6  # ms one-way
            latency_filter.update(rtt)
        except BlockingIOError:
            pass
        time.sleep(0.2)                  # 5 Hz

if latency_filter.mean_ms > 80 or probe_loss_rate > 0.05:
    safe_mode.activate()
