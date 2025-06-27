import json
import logging
import socket
import time
import threading
import queue
import struct
import pickle
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
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(name)s - %(message)s', 
                                    datefmt='%Y-%m-%d %H:%M:%S')
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
    """Lightweight, fire-and-forget UDP sender used by the teleoperation *leader* machine.

    This class is intentionally minimal: it serialises action dictionaries to JSON and pushes them
    over UDP to the *server* (robot/follower machine) with best-effort delivery. If packets are lost
    they are simply dropped – the next action will arrive a few milliseconds later anyway.
    """

    def __init__(self, server: str, log_file: str = "udp_sender.log", async_logging: bool = True, use_binary: bool = True, enable_logging: bool = False):
        """Args
        ----
        server: str
            Remote endpoint in the form "<ip>:<port>" (e.g. "10.10.10.10:5555").
        log_file: str
            Path to log file for tracking send timestamps.
        async_logging: bool
            Whether to use async logging to minimize latency impact.
        use_binary: bool
            Whether to use binary format instead of JSON for maximum performance.
        enable_logging: bool
            Whether to enable logging at all. Set to False for minimum latency.
        """
        ip, port_str = server.split(":")
        self._addr: Tuple[str, int] = (ip, int(port_str))
        self._use_binary = use_binary
        self._enable_logging = enable_logging

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Non-blocking send – we never wait for ACKs.
        self._sock.setblocking(False)

        # Packet sequence number for identification
        self._sequence_number = 0

        # Setup logging only if enabled
        if self._enable_logging:
            self._logger = logging.getLogger(f"UDPSender_{ip}_{port_str}")
            self._logger.setLevel(logging.INFO)
            
            # Create file handler if not already exists
            if not self._logger.handlers:
                if async_logging:
                    handler = AsyncLogHandler(log_file)
                else:
                    handler = logging.FileHandler(log_file)
                    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(name)s - %(message)s', 
                                                datefmt='%Y-%m-%d %H:%M:%S')
                    handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.propagate = False
        else:
            self._logger = None

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def send(self, action: Dict[str, float]) -> None:  # noqa: D401 – present tense OK
        """Serialise *action* to JSON and transmit it over UDP."""
        send_timestamp = time.time()
        
        if self._use_binary:
            # Binary format for maximum performance
            payload = self._serialize_binary(action, send_timestamp)
        else:
            # JSON format for compatibility
            packet_data = {
                "_packet_id": self._sequence_number,
                "_send_timestamp": send_timestamp,
                "action": action
            }
            payload = json.dumps(packet_data).encode("utf-8")
        
        # Best-effort – if the socket is not ready we'll simply drop the frame.
        try:
            self._sock.sendto(payload, self._addr)
            if self._enable_logging and self._logger:
                self._logger.info(f"SENT - packet_id: {self._sequence_number}, timestamp: {send_timestamp:.6f}, payload_size: {len(payload)} bytes, action: {action}")
        except (BlockingIOError, OSError) as e:
            # Dropped – nothing to do, next frame will go through.
            if self._enable_logging and self._logger:
                self._logger.warning(f"DROPPED - packet_id: {self._sequence_number}, timestamp: {send_timestamp:.6f}, error: {e}, payload_size: {len(payload)} bytes, action: {action}")
        
        # Increment sequence number for next packet
        self._sequence_number += 1
    
    def _serialize_binary(self, action: Dict[str, float], timestamp: float) -> bytes:
        """Serialize action to binary format for maximum performance."""
        # Pre-calculate total size to avoid memory reallocations
        total_size = 20  # header: magic(4) + seq(4) + timestamp(8) + count(4)
        key_sizes = []
        for key in action:
            key_bytes = key.encode('utf-8')
            key_sizes.append((key, key_bytes))
            total_size += 4 + len(key_bytes) + 8  # key_len(4) + key + value(8)
        
        # Allocate buffer once
        payload = bytearray(total_size)
        offset = 0
        
        # Write header
        payload[offset:offset+4] = b'UDPA'  # magic
        offset += 4
        struct.pack_into('<I', payload, offset, self._sequence_number)
        offset += 4
        struct.pack_into('<d', payload, offset, timestamp)
        offset += 8
        struct.pack_into('<I', payload, offset, len(action))
        offset += 4
        
        # Write action data
        for key, key_bytes in key_sizes:
            struct.pack_into('<I', payload, offset, len(key_bytes))
            offset += 4
            payload[offset:offset+len(key_bytes)] = key_bytes
            offset += len(key_bytes)
            struct.pack_into('<d', payload, offset, action[key])
            offset += 8
        
        return bytes(payload)


class UDPTransportReceiver:
    """Blocking UDP receiver used by the teleoperation *follower* machine."""

    def __init__(self, port: int, buffer_size: int = 65535, log_file: str = "udp_receiver.log", async_logging: bool = True, use_binary: bool = True, enable_logging: bool = False):
        """Listen on *port* for incoming action packets.
        
        Args
        ----
        port: int
            Port to listen on for incoming packets.
        buffer_size: int
            Maximum size of UDP packets to receive.
        log_file: str
            Path to log file for tracking receive timestamps.
        async_logging: bool
            Whether to use async logging to minimize latency impact.
        use_binary: bool
            Whether to use binary format instead of JSON for maximum performance.
        enable_logging: bool
            Whether to enable logging at all. Set to False for minimum latency.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow immediate rebinding after a restart.
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind on all interfaces – users can rely on firewall rules for protection.
        self._sock.bind(("", port))
        self._buffer_size = buffer_size
        self._use_binary = use_binary
        self._enable_logging = enable_logging

        # Track last received packet ID to detect gaps
        self._last_packet_id = -1

        # Setup logging only if enabled
        if self._enable_logging:
            self._logger = logging.getLogger(f"UDPReceiver_{port}")
            self._logger.setLevel(logging.INFO)
            
            # Create file handler if not already exists
            if not self._logger.handlers:
                if async_logging:
                    handler = AsyncLogHandler(log_file)
                else:
                    handler = logging.FileHandler(log_file)
                    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(name)s - %(message)s', 
                                                datefmt='%Y-%m-%d %H:%M:%S')
                    handler.setFormatter(formatter)
                self._logger.addHandler(handler)
                self._logger.propagate = False
        else:
            self._logger = None

    # ------------------------------------------------------------------
    def recv(self) -> Dict[str, float]:  # noqa: D401
        """Block until the next action packet is received and return it."""
        payload, addr = self._sock.recvfrom(self._buffer_size)
        recv_timestamp = time.time()
        
        try:
            if self._use_binary:
                action, packet_id, send_timestamp = self._deserialize_binary(payload)
            else:
                action, packet_id, send_timestamp = self._deserialize_json(payload)
            
            # Calculate latency if send timestamp is available
            latency_ms = (recv_timestamp - send_timestamp) * 1000 if send_timestamp else 0
            
            # Check for missing packets
            if packet_id is not None and self._last_packet_id >= 0:
                expected_id = self._last_packet_id + 1
                if packet_id != expected_id:
                    if packet_id > expected_id:
                        missing_count = packet_id - expected_id
                        if self._enable_logging and self._logger:
                            self._logger.warning(f"PACKET_LOSS - missing {missing_count} packet(s), expected: {expected_id}, received: {packet_id}")
                    else:
                        if self._enable_logging and self._logger:
                            self._logger.warning(f"OUT_OF_ORDER - received: {packet_id}, expected: {expected_id}")
            
            if packet_id is not None:
                self._last_packet_id = packet_id
                if self._enable_logging and self._logger:
                    self._logger.info(f"RECEIVED - packet_id: {packet_id}, recv_timestamp: {recv_timestamp:.6f}, send_timestamp: {send_timestamp:.6f}, latency: {latency_ms:.2f}ms, from: {addr[0]}:{addr[1]}, payload_size: {len(payload)} bytes, action: {action}")
            else:
                if self._enable_logging and self._logger:
                    self._logger.info(f"RECEIVED - legacy_format, timestamp: {recv_timestamp:.6f}, from: {addr[0]}:{addr[1]}, payload_size: {len(payload)} bytes, action: {action}")
            
            return action
            
        except (json.JSONDecodeError, struct.error, UnicodeDecodeError) as e:
            if self._enable_logging and self._logger:
                self._logger.error(f"DECODE_ERROR - timestamp: {recv_timestamp:.6f}, from: {addr[0]}:{addr[1]}, error: {e}, payload: {payload[:100]}...")
            raise
    
    def _deserialize_json(self, payload: bytes) -> Tuple[Dict[str, float], Optional[int], float]:
        """Deserialize JSON payload."""
        packet_data = json.loads(payload.decode("utf-8"))
        
        # Handle both new format (with packet metadata) and legacy format (raw action)
        if isinstance(packet_data, dict) and "_packet_id" in packet_data:
            # New format with packet identification
            packet_id = packet_data["_packet_id"]
            send_timestamp = packet_data.get("_send_timestamp", 0)
            action = packet_data["action"]
            return action, packet_id, send_timestamp
        else:
            # Legacy format (raw action dictionary) - for backward compatibility
            action = packet_data
            return action, None, 0
    
    def _deserialize_binary(self, payload: bytes) -> Tuple[Dict[str, float], int, float]:
        """Deserialize binary payload for maximum performance."""
        # Format: [magic:4][seq:4][timestamp:8][action_count:4][key1_len:4][key1:str][val1:8][key2_len:4][key2:str][val2:8]...
        if len(payload) < 20:  # Minimum size for header
            raise struct.error("Payload too short")
        
        magic = payload[:4]
        if magic != b'UDPA':
            raise struct.error("Invalid magic number")
        
        packet_id = struct.unpack('<I', payload[4:8])[0]
        send_timestamp = struct.unpack('<d', payload[8:16])[0]
        action_count = struct.unpack('<I', payload[16:20])[0]
        
        action = {}
        offset = 20
        
        for _ in range(action_count):
            if offset + 4 > len(payload):
                raise struct.error("Payload truncated")
            
            key_len = struct.unpack('<I', payload[offset:offset+4])[0]
            offset += 4
            
            if offset + key_len + 8 > len(payload):
                raise struct.error("Payload truncated")
            
            key = payload[offset:offset+key_len].decode('utf-8')
            offset += key_len
            
            value = struct.unpack('<d', payload[offset:offset+8])[0]
            offset += 8
            
            action[key] = value
        
        return action, packet_id, send_timestamp 