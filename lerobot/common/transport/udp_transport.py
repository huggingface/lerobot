import json
import logging
import socket
import time
import threading
import queue
import struct
import pickle
import select
import gc
from typing import Dict, Tuple, Optional

__all__ = [
    "UDPTransportSender",
    "UDPTransportReceiver",
]


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler to prevent file I/O from blocking the main thread."""
    
    def __init__(self, log_file: str, max_queue_size: int = 1000, max_file_size: int = 10*1024*1024):  # 10MB
        super().__init__()
        self._queue = queue.Queue(maxsize=max_queue_size)
        self._log_file = log_file
        self._max_file_size = max_file_size
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
                
                # Check file size and rotate if needed
                try:
                    if self._handler.stream.tell() > self._max_file_size:
                        self._rotate_log_file()
                except (OSError, AttributeError):
                    pass  # File might not be seekable
                
                self._handler.emit(record)
            except Exception:
                pass  # Don't let logging errors crash the worker
    
    def _rotate_log_file(self):
        """Rotate the log file to prevent it from growing too large."""
        try:
            self._handler.close()
            import os
            if os.path.exists(self._log_file):
                backup_file = f"{self._log_file}.old"
                if os.path.exists(backup_file):
                    os.remove(backup_file)
                os.rename(self._log_file, backup_file)
            
            self._handler = logging.FileHandler(self._log_file)
            formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(name)s - %(message)s', 
                                        datefmt='%Y-%m-%d %H:%M:%S')
            self._handler.setFormatter(formatter)
        except Exception:
            pass  # Don't let rotation errors crash the worker
    
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

    def __init__(self, server: str, log_file: str = "udp_sender.log", async_logging: bool = True, use_binary: bool = True, enable_logging: bool = False, ultra_fast: bool = True):
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
        ultra_fast: bool
            Whether to use ultra-fast mode with minimal overhead.
        """
        ip, port_str = server.split(":")
        self._addr: Tuple[str, int] = (ip, int(port_str))
        self._use_binary = use_binary
        self._enable_logging = enable_logging
        self._ultra_fast = ultra_fast

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
        
        if self._ultra_fast:
            # Ultra-fast format for maximum performance
            payload = self._serialize_ultra_fast(action, send_timestamp)
        elif self._use_binary:
            # Binary format for compatibility
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
    
    def _serialize_ultra_fast(self, action: Dict[str, float], timestamp: float) -> bytes:
        """Ultra-fast serialization with minimal overhead."""
        # Pre-calculate total size to avoid memory reallocations
        total_size = 9  # header: timestamp(8) + count(1)
        key_sizes = []
        for key in action:
            key_bytes = key.encode('ascii')
            key_sizes.append((key, key_bytes))
            total_size += 1 + len(key_bytes) + 4  # key_len(1) + key + value(4)
        
        # Allocate buffer once
        payload = bytearray(total_size)
        offset = 0
        
        # Write header
        struct.pack_into('<d', payload, offset, timestamp)
        offset += 8
        payload[offset] = len(action)
        offset += 1
        
        # Write action data
        for key, key_bytes in key_sizes:
            payload[offset] = len(key_bytes)
            offset += 1
            payload[offset:offset+len(key_bytes)] = key_bytes
            offset += len(key_bytes)
            struct.pack_into('<f', payload, offset, action[key])
            offset += 4
        
        return bytes(payload)
    
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

    def __init__(self, port: int, buffer_size: int = 65535, log_file: str = "udp_receiver.log", async_logging: bool = True, use_binary: bool = True, enable_logging: bool = True, drop_old_packets: bool = True, ultra_fast: bool = True):
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
        drop_old_packets: bool
            Whether to drop old packets when buffer is full to maintain low latency.
        ultra_fast: bool
            Whether to use ultra-fast mode with minimal overhead.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow immediate rebinding after a restart.
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        # Increase receive buffer size significantly
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 262144)  # 256KB
        
        # Bind on all interfaces – users can rely on firewall rules for protection.
        self._sock.bind(("", port))
        self._buffer_size = buffer_size
        self._use_binary = use_binary
        self._enable_logging = enable_logging
        self._drop_old_packets = drop_old_packets
        self._ultra_fast = ultra_fast

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
        if self._ultra_fast:
            # Ultra-fast path with minimal overhead
            payload, addr = self._sock.recvfrom(self._buffer_size)
            recv_timestamp = time.time()
            
            try:
                action, send_timestamp = self._deserialize_ultra_fast(payload)
                
                # Add basic logging if enabled (minimal overhead)
                if self._enable_logging and self._logger:
                    latency_ms = (recv_timestamp - send_timestamp) * 1000 if send_timestamp else 0
                    self._logger.info(f"RECEIVED_ULTRA - latency: {latency_ms:.2f}ms")
                
                return action
            except (struct.error, UnicodeDecodeError) as e:
                if self._enable_logging and self._logger:
                    self._logger.error(f"ULTRA_DECODE_ERROR - error: {e}")
                return {}
        
        # Use select with timeout to prevent indefinite blocking
        ready = select.select([self._sock], [], [], 0.1)  # 100ms timeout
        if not ready[0]:
            # No data available, return empty action to prevent blocking
            if self._enable_logging and self._logger:
                self._logger.warning("TIMEOUT - No data received within 100ms")
            return {}
        
        payload, addr = self._sock.recvfrom(self._buffer_size)
        recv_timestamp = time.time()
        
        try:
            if self._use_binary:
                action, packet_id, send_timestamp = self._deserialize_binary(payload)
            else:
                action, packet_id, send_timestamp = self._deserialize_json(payload)
            
            # Calculate latency if send timestamp is available
            latency_ms = (recv_timestamp - send_timestamp) * 1000 if send_timestamp else 0
            
            # Drop old packets if latency is too high
            if self._drop_old_packets and latency_ms > 500:  # Drop packets older than 500ms
                if self._enable_logging and self._logger:
                    self._logger.warning(f"DROPPED_OLD - packet_id: {packet_id}, latency: {latency_ms:.2f}ms")
                # Try to get a fresher packet by draining the buffer
                return self._recv_fresh_packet()
            
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
            
            return action
            
        except (json.JSONDecodeError, struct.error, UnicodeDecodeError) as e:
            if self._enable_logging and self._logger:
                self._logger.error(f"DECODE_ERROR - timestamp: {recv_timestamp:.6f}, from: {addr[0]}:{addr[1]}, error: {e}, payload: {payload[:100]}...")
            raise
    
    def _deserialize_ultra_fast(self, payload: bytes) -> Tuple[Dict[str, float], float]:
        """Ultra-fast deserialization with minimal overhead."""
        # Format: [timestamp:8][count:1][key1_len:1][key1:str][val1:4][key2_len:1][key2:str][val2:4]...
        if len(payload) < 9:
            raise struct.error("Payload too short")
        
        send_timestamp = struct.unpack('<d', payload[:8])[0]
        action_count = payload[8]
        
        action = {}
        offset = 9
        
        for _ in range(action_count):
            if offset + 1 > len(payload):
                raise struct.error("Payload truncated")
            
            key_len = payload[offset]
            offset += 1
            
            if offset + key_len + 4 > len(payload):
                raise struct.error("Payload truncated")
            
            key = payload[offset:offset+key_len].decode('ascii')
            offset += key_len
            
            value = struct.unpack('<f', payload[offset:offset+4])[0]
            offset += 4
            
            action[key] = value
        
        return action, send_timestamp
    
    def _recv_fresh_packet(self) -> Dict[str, float]:
        """Drain the receive buffer to get the freshest packet."""
        # Temporarily set non-blocking mode
        self._sock.setblocking(False)
        
        latest_payload = None
        latest_addr = None
        latest_timestamp = 0
        
        # Drain all available packets and keep the freshest one
        try:
            while True:
                ready = select.select([self._sock], [], [], 0.001)  # 1ms timeout
                if not ready[0]:
                    break
                
                payload, addr = self._sock.recvfrom(self._buffer_size)
                recv_timestamp = time.time()
                
                try:
                    if self._use_binary:
                        action, packet_id, send_timestamp = self._deserialize_binary(payload)
                    else:
                        action, packet_id, send_timestamp = self._deserialize_json(payload)
                    
                    # Keep the packet with the lowest latency
                    latency_ms = (recv_timestamp - send_timestamp) * 1000 if send_timestamp else 0
                    if latency_ms < latest_timestamp or latest_payload is None:
                        latest_payload = payload
                        latest_addr = addr
                        latest_timestamp = latency_ms
                        
                except (json.JSONDecodeError, struct.error, UnicodeDecodeError):
                    continue  # Skip malformed packets
                    
        except (BlockingIOError, OSError):
            pass  # No more data available
        
        # Restore blocking mode
        self._sock.setblocking(True)
        
        # Process the freshest packet
        if latest_payload is not None:
            try:
                if self._use_binary:
                    action, packet_id, send_timestamp = self._deserialize_binary(latest_payload)
                else:
                    action, packet_id, send_timestamp = self._deserialize_json(latest_payload)
                
                if self._enable_logging and self._logger:
                    latency_ms = (time.time() - send_timestamp) * 1000 if send_timestamp else 0
                    self._logger.info(f"FRESH_PACKET - packet_id: {packet_id}, latency: {latency_ms:.2f}ms")
                
                return action
            except (json.JSONDecodeError, struct.error, UnicodeDecodeError):
                pass
        
        # If no valid packet found, return empty action
        return {}
    
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