import grpc
import time
import threading
import numpy as np
from concurrent import futures
from queue import Queue, Empty
from typing import Optional, Union

import async_inference_pb2  # type: ignore
import async_inference_pb2_grpc  # type: ignore

class RobotClient:
    def __init__(self, server_address="localhost:50051"):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = async_inference_pb2_grpc.AsyncInferenceStub(self.channel)
        
        self.running = False
        self.first_observation_sent = False
        self.action_chunk_size = 10

        self.action_queue = Queue()
        self.action_queue_lock = threading.Lock()

        # debugging purposes
        self.action_buffer = []
        
    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # client-server handshake
            self.stub.Ready(async_inference_pb2.Empty())
            print("Connected to policy server")
            
            self.running = True
            return True
        
        except grpc.RpcError as e:
            print(f"Failed to connect to policy server: {e}")
            return False
    
    def stop(self):
        """Stop the robot client"""
        self.running = False
        self.channel.close()
        
    def send_observation(
            self, 
            observation_data: Union[np.ndarray, bytes], 
            transfer_state: async_inference_pb2.TransferState = async_inference_pb2.TRANSFER_MIDDLE
        ) -> bool:
        """Send observation to the policy server. 
        Returns True if the observation was sent successfully, False otherwise."""
        if not self.running:
            print("Client not running")
            return False

        # Convert observation data to bytes
        if not isinstance(observation_data, bytes):
            observation_data = np.array(observation_data).tobytes()

        observation = async_inference_pb2.Observation(
            transfer_state=transfer_state,
            data=observation_data
        )

        try:
            _ = self.stub.SendObservations(iter([observation]))
            if transfer_state == async_inference_pb2.TRANSFER_BEGIN:
                self.first_observation_sent = True
            return True

        except grpc.RpcError as e:
            print(f"Error sending observation: {e}")
            return False
    
    def _should_replace_queue(self, percentage_left: float = 0.5) -> bool:
        """Check if we should replace the queue based on consumption rate"""
        with self.action_queue_lock:
            current_size = self.action_queue.qsize()
            return current_size/self.action_chunk_size <= percentage_left

    def _clear_and_refill_queue(self, actions: list[np.ndarray]):
        """Clear the existing queue and fill it with new actions"""
        assert len(actions) == self.action_chunk_size, \
            f"Action batch size must match action chunk!" \
            f"size: {len(actions)} != {self.action_chunk_size}"
        
        with self.action_queue_lock:
            # Clear the queue
            while not self.action_queue.empty():
                try:
                    self.action_queue.get_nowait()
                except Empty:
                    break
            
            # Fill with new actions
            for action in actions:
                self.action_queue.put(action)


    def receive_actions(self):
        """Receive actions from the policy server"""
        while self.running:
            # Wait until first observation is sent
            if not self.first_observation_sent:
                time.sleep(0.1)
                continue
                        
            try:
                # Use StreamActions to get a stream of actions from the server
                action_batch = []
                for action in self.stub.StreamActions(async_inference_pb2.Empty()):
                    # NOTE: reading from buffer with numpy requires reshaping
                    action_data = np.frombuffer(
                        action.data, dtype=np.float32
                    ).reshape(self.action_chunk_size, -1)

                    for a in action_data:
                        action_batch.append(a)
                    
                # Replace entire queue with new batch of actions
                if action_batch and self._should_replace_queue():
                    self._clear_and_refill_queue(action_batch)
                        
            except grpc.RpcError as e:
                print(f"Error receiving actions: {e}")
                time.sleep(1)  # Avoid tight loop on error
    
    def get_next_action(self) -> Optional[np.ndarray]:
        """Get the next action from the queue"""
        try:
            with self.action_queue_lock:
                return self.action_queue.get_nowait()
        except Empty:
            return None
    
    def stream_observations(self, get_observation_fn):
        """Continuously stream observations to the server"""
        first_observation = True
        while self.running:
            try:
                observation = get_observation_fn()
                
                # Set appropriate transfer state
                if first_observation:
                    state = async_inference_pb2.TRANSFER_BEGIN
                    first_observation = False
                else:
                    state = async_inference_pb2.TRANSFER_MIDDLE

                self.send_observation(observation, state)
                time.sleep(0.1)  # Adjust rate as needed
                
            except Exception as e:
                print(f"Error in observation sender: {e}")
                time.sleep(1)

def example_usage():
    # Example of how to use the RobotClient
    client = RobotClient()
    
    if client.start():
        # Function to generate mock observations
        def get_mock_observation():
            return np.random.randint(0, 10, size=10).astype(np.float32)
        
        # Create and start observation sender thread
        obs_thread = threading.Thread(
            target=client.stream_observations,
            args=(get_mock_observation,)
        )
        obs_thread.daemon = True
        obs_thread.start()
        
        # Create and start action receiver thread
        action_thread = threading.Thread(target=client.receive_actions)
        action_thread.daemon = True
        action_thread.start()
        
        try:
            # Main loop - action execution
            while True:
                print(client.action_queue.qsize())
                action = client.get_next_action()
                if action is not None:
                    print(f"Executing action: {action}")
                    time.sleep(1)
                else:
                    print("No action available")
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            pass
        
        finally:
            client.stop()

if __name__ == "__main__":
    example_usage()

