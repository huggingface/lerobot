import grpc
import time
import threading
import numpy as np
from concurrent import futures

import async_inference_pb2  # type: ignore
import async_inference_pb2_grpc  # type: ignore

class RobotClient:
    def __init__(self, server_address="localhost:50051"):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = async_inference_pb2_grpc.AsyncInferenceStub(self.channel)
        self.running = False
        self.action_callback = None
        
    def start(self):
        """Start the robot client and connect to the policy server"""
        try:
            # Check if the server is ready
            self.stub.Ready(async_inference_pb2.Empty())
            print("Connected to policy server server")
            self.running = True
            
            # Start action receiving thread
            self.action_thread = threading.Thread(target=self.receive_actions)
            self.action_thread.daemon = True
            self.action_thread.start()
            
            return True
        except grpc.RpcError as e:
            print(f"Failed to connect to policy server: {e}")
            return False
    
    def stop(self):
        """Stop the robot client"""
        self.running = False
        self.channel.close()
        
    def send_observation(self, observation_data, transfer_state=async_inference_pb2.TRANSFER_MIDDLE):
        """Send a single observation to the policy server"""
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
            # For a single observation
            response_future = self.stub.SendObservations(iter([observation]))
            return True
        except grpc.RpcError as e:
            print(f"Error sending observation: {e}")
            return False
            
    
    def receive_actions(self):
        """Receive actions from the policy server"""
        while self.running:
            try:
                # Use StreamActions to get a stream of actions from the server
                for action in self.stub.StreamActions(async_inference_pb2.Empty()):
                    if self.action_callback:
                        # Convert bytes back to data (assuming numpy array)
                        action_data = np.frombuffer(action.data)
                        self.action_callback(
                            action_data, 
                            action.transfer_state
                        )
                    else:
                        print(
                            "Received action: ",
                            f"state={action.transfer_state}, ",
                            f"data size={len(action.data)} bytes"
                        )
            
            except grpc.RpcError as e:
                print(f"Error receiving actions: {e}")
                time.sleep(1)  # Avoid tight loop on error
    
    def register_action_callback(self, callback):
        """Register a callback for when actions are received"""
        self.action_callback = callback
    

def example_usage():
    # Example of how to use the RobotClient
    client = RobotClient()
    
    if client.start():
        # Define a callback for received actions
        def on_action(action_data, transfer_state):
            print(f"Action received: state={transfer_state}, data={action_data[:10]}...")
        
        client.register_action_callback(on_action)
        
        # Send some example observations
        for i in range(10):
            # Create dummy observation data
            observation = np.arange(10, dtype=np.float32)
            
            # Send it to the policy server
            if i == 0:
                state = async_inference_pb2.TRANSFER_BEGIN
            elif i == 9:
                state = async_inference_pb2.TRANSFER_END
            else:
                state = async_inference_pb2.TRANSFER_MIDDLE
                
            client.send_observation(observation, state)
            print(f"Sent observation {i+1}/10")
            time.sleep(0.5)
        
        # Keep the main thread alive to receive actions
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            client.stop()

if __name__ == "__main__":
    example_usage()

