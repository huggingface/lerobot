import json
import logging
import socket
import time
from typing import Dict, Tuple

__all__ = [
    "UDPTransportSender",
    "UDPTransportReceiver",
]


class UDPTransportSender:
    """Lightweight, fire-and-forget UDP sender used by the teleoperation *leader* machine.

    This class is intentionally minimal: it serialises action dictionaries to JSON and pushes them
    over UDP to the *server* (robot/follower machine) with best-effort delivery. If packets are lost
    they are simply dropped – the next action will arrive a few milliseconds later anyway.
    """

    def __init__(self, server: str, log_file: str = "udp_sender.log"):
        """Args
        ----
        server: str
            Remote endpoint in the form "<ip>:<port>" (e.g. "10.10.10.10:5555").
        log_file: str
            Path to log file for tracking send timestamps.
        """
        ip, port_str = server.split(":")
        self._addr: Tuple[str, int] = (ip, int(port_str))

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Non-blocking send – we never wait for ACKs.
        self._sock.setblocking(False)

        # Packet sequence number for identification
        self._sequence_number = 0

        # Setup logging
        self._logger = logging.getLogger(f"UDPSender_{ip}_{port_str}")
        self._logger.setLevel(logging.INFO)
        
        # Create file handler if not already exists
        if not self._logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(name)s - %(message)s', 
                                        datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.propagate = False

    # ---------------------------------------------------------------------
    # Public helpers
    # ---------------------------------------------------------------------
    def send(self, action: Dict[str, float]) -> None:  # noqa: D401 – present tense OK
        """Serialise *action* to JSON and transmit it over UDP."""
        send_timestamp = time.time()
        
        # Add packet identification and timestamp to the payload
        packet_data = {
            "_packet_id": self._sequence_number,
            "_send_timestamp": send_timestamp,
            "action": action
        }
        
        payload = json.dumps(packet_data).encode("utf-8")
        
        # Best-effort – if the socket is not ready we'll simply drop the frame.
        try:
            self._sock.sendto(payload, self._addr)
            self._logger.info(f"SENT - packet_id: {self._sequence_number}, timestamp: {send_timestamp:.6f}, payload_size: {len(payload)} bytes, action_keys: {list(action.keys())}")
        except (BlockingIOError, OSError) as e:
            # Dropped – nothing to do, next frame will go through.
            self._logger.warning(f"DROPPED - packet_id: {self._sequence_number}, timestamp: {send_timestamp:.6f}, error: {e}, payload_size: {len(payload)} bytes")
        
        # Increment sequence number for next packet
        self._sequence_number += 1


class UDPTransportReceiver:
    """Blocking UDP receiver used by the teleoperation *follower* machine."""

    def __init__(self, port: int, buffer_size: int = 65535, log_file: str = "udp_receiver.log"):
        """Listen on *port* for incoming action packets.
        
        Args
        ----
        port: int
            Port to listen on for incoming packets.
        buffer_size: int
            Maximum size of UDP packets to receive.
        log_file: str
            Path to log file for tracking receive timestamps.
        """
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Allow immediate rebinding after a restart.
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind on all interfaces – users can rely on firewall rules for protection.
        self._sock.bind(("", port))
        self._buffer_size = buffer_size

        # Track last received packet ID to detect gaps
        self._last_packet_id = -1

        # Setup logging
        self._logger = logging.getLogger(f"UDPReceiver_{port}")
        self._logger.setLevel(logging.INFO)
        
        # Create file handler if not already exists
        if not self._logger.handlers:
            handler = logging.FileHandler(log_file)
            formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(name)s - %(message)s', 
                                        datefmt='%Y-%m-%d %H:%M:%S')
            handler.setFormatter(formatter)
            self._logger.addHandler(handler)
            self._logger.propagate = False

    # ------------------------------------------------------------------
    def recv(self) -> Dict[str, float]:  # noqa: D401
        """Block until the next action packet is received and return it."""
        payload, addr = self._sock.recvfrom(self._buffer_size)
        recv_timestamp = time.time()
        
        try:
            packet_data = json.loads(payload.decode("utf-8"))
            
            # Handle both new format (with packet metadata) and legacy format (raw action)
            if isinstance(packet_data, dict) and "_packet_id" in packet_data:
                # New format with packet identification
                packet_id = packet_data["_packet_id"]
                send_timestamp = packet_data.get("_send_timestamp", 0)
                action = packet_data["action"]
                
                # Calculate latency if send timestamp is available
                latency_ms = (recv_timestamp - send_timestamp) * 1000 if send_timestamp else 0
                
                # Check for missing packets
                if self._last_packet_id >= 0:
                    expected_id = self._last_packet_id + 1
                    if packet_id != expected_id:
                        if packet_id > expected_id:
                            missing_count = packet_id - expected_id
                            self._logger.warning(f"PACKET_LOSS - missing {missing_count} packet(s), expected: {expected_id}, received: {packet_id}")
                        else:
                            self._logger.warning(f"OUT_OF_ORDER - received: {packet_id}, expected: {expected_id}")
                
                self._last_packet_id = packet_id
                
                self._logger.info(f"RECEIVED - packet_id: {packet_id}, recv_timestamp: {recv_timestamp:.6f}, send_timestamp: {send_timestamp:.6f}, latency: {latency_ms:.2f}ms, from: {addr[0]}:{addr[1]}, payload_size: {len(payload)} bytes, action_keys: {list(action.keys())}")
            else:
                # Legacy format (raw action dictionary) - for backward compatibility
                action = packet_data
                self._logger.info(f"RECEIVED - legacy_format, timestamp: {recv_timestamp:.6f}, from: {addr[0]}:{addr[1]}, payload_size: {len(payload)} bytes, action_keys: {list(action.keys())}")
            
            return action
            
        except json.JSONDecodeError as e:
            self._logger.error(f"JSON_ERROR - timestamp: {recv_timestamp:.6f}, from: {addr[0]}:{addr[1]}, error: {e}, payload: {payload[:100]}...")
            raise 