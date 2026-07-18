# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Multi-Policy Server that launches multiple policy servers on different ports.

Example:
```shell
python src/lerobot/scripts/server/multi_policy_server.py \
     --host=127.0.0.1 \
     --base_port=8080 \
     --num_servers=4 \
     --fps=30 \
     --inference_latency=0.033 \
     --obs_queue_timeout=1
```

This will launch 4 servers on ports 8080, 8081, 8082, 8083.
"""

import logging
import multiprocessing
import signal
import sys
import threading
import time
from concurrent import futures
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import List

import draccus
import grpc

from lerobot.scripts.server.configs import PolicyServerConfig
from lerobot.scripts.server.policy_server import PolicyServer
from lerobot.transport import services_pb2_grpc


@dataclass
class MultiPolicyServerConfig:
    # Host to bind the servers to
    host: str = "127.0.0.1"
    # Base port number (servers will use base_port, base_port+1, base_port+2, etc.)
    base_port: int = 8080
    # Number of policy servers to launch
    num_servers: int = 4
    # Frames per second for each server
    fps: int = 30
    # Simulated inference latency in seconds
    inference_latency: float = 0.033
    # Timeout for observation queue
    obs_queue_timeout: float = 1.0


class MultiPolicyServerManager:
    """Manager class for multiple policy servers."""
    
    def __init__(self, config: MultiPolicyServerConfig):
        self.config = config
        self.servers = []
        self.policy_servers = []
        self.processes = []
        self.shutdown_event = threading.Event()
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("MultiPolicyServer")
        
    def _create_server_process(self, port: int) -> multiprocessing.Process:
        """Create a server process for a specific port."""
        
        def run_server(port: int, config: MultiPolicyServerConfig):
            """Function to run a single policy server in a separate process."""
            # Create policy server config for this specific port
            server_config = PolicyServerConfig(
                host=config.host,
                port=port,
                fps=config.fps,
                inference_latency=config.inference_latency,
                obs_queue_timeout=config.obs_queue_timeout
            )
            
            # Setup logging for this process
            logging.basicConfig(
                level=logging.INFO,
                format=f'%(asctime)s - Server-{port} - %(name)s - %(levelname)s - %(message)s'
            )
            logger = logging.getLogger(f"PolicyServer-{port}")
            
            try:
                # Create the policy server instance
                policy_server = PolicyServer(server_config)
                
                # Setup and start gRPC server
                server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
                services_pb2_grpc.add_AsyncInferenceServicer_to_server(policy_server, server)
                server.add_insecure_port(f"{config.host}:{port}")
                
                logger.info(f"PolicyServer started on {config.host}:{port}")
                server.start()
                
                # Keep the server running
                try:
                    server.wait_for_termination()
                except KeyboardInterrupt:
                    logger.info(f"Server on port {port} received shutdown signal")
                finally:
                    policy_server.stop()
                    server.stop(grace=5)
                    logger.info(f"Server on port {port} terminated")
                    
            except Exception as e:
                logger.error(f"Error starting server on port {port}: {e}")
                
        return multiprocessing.Process(target=run_server, args=(port, self.config))
    
    def start_servers(self):
        """Start all policy servers."""
        self.logger.info(f"Starting {self.config.num_servers} policy servers...")
        
        # Create and start a process for each server
        for i in range(self.config.num_servers):
            port = self.config.base_port + i
            process = self._create_server_process(port)
            process.start()
            self.processes.append(process)
            self.logger.info(f"Started server {i+1}/{self.config.num_servers} on port {port}")
            
        self.logger.info("All servers started successfully!")
        self.logger.info("Server ports:")
        for i in range(self.config.num_servers):
            port = self.config.base_port + i
            self.logger.info(f"  Server {i+1}: {self.config.host}:{port}")
            
    def stop_servers(self):
        """Stop all policy servers."""
        self.logger.info("Stopping all servers...")
        self.shutdown_event.set()
        
        # Terminate all processes
        for i, process in enumerate(self.processes):
            if process.is_alive():
                self.logger.info(f"Terminating server process {i+1}")
                process.terminate()
                
        # Wait for all processes to finish
        for i, process in enumerate(self.processes):
            process.join(timeout=10)
            if process.is_alive():
                self.logger.warning(f"Force killing server process {i+1}")
                process.kill()
                process.join()
                
        self.logger.info("All servers stopped")
        
    def wait_for_shutdown(self):
        """Wait for shutdown signal and keep servers running."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down...")
            self.stop_servers()
            sys.exit(0)
            
        # Register signal handlers
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Keep the main thread alive
            while not self.shutdown_event.is_set():
                # Check if any process has died unexpectedly
                for i, process in enumerate(self.processes):
                    if not process.is_alive():
                        self.logger.error(f"Server process {i+1} died unexpectedly")
                        
                time.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
            self.stop_servers()


@draccus.wrap()
def serve_multi(cfg: MultiPolicyServerConfig):
    """Start multiple policy servers with the given configuration.

    Args:
        cfg: MultiPolicyServerConfig instance containing server configuration.
    """
    logging.info("Multi-Policy Server Configuration:")
    logging.info(pformat(asdict(cfg)))
    
    # Validate configuration
    if cfg.num_servers <= 0:
        raise ValueError("Number of servers must be positive")
        
    if cfg.base_port <= 0 or cfg.base_port > 65535:
        raise ValueError("Base port must be between 1 and 65535")
        
    if cfg.base_port + cfg.num_servers - 1 > 65535:
        raise ValueError(f"Port range exceeds maximum (base_port + num_servers - 1 = {cfg.base_port + cfg.num_servers - 1})")
    
    # Create and start the multi-server manager
    manager = MultiPolicyServerManager(cfg)
    
    try:
        manager.start_servers()
        manager.wait_for_shutdown()
    except Exception as e:
        logging.error(f"Error in multi-policy server: {e}")
        manager.stop_servers()
        raise


if __name__ == "__main__":
    serve_multi()
