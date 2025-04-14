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
                    action_data = np.frombuffer(action.data, dtype=np.float32)
                    print(
                        "Received action: ",
                        f"state={action.transfer_state}, ",
                        f"data={action_data}, ",
                        f"data size={len(action.data)} bytes"
                    )

            except grpc.RpcError as e:
                print(f"Error receiving actions: {e}")
                time.sleep(1)  # Avoid tight loop on error


def example_usage():
    # Example of how to use the RobotClient
    client = RobotClient()
    
    if client.start():
        # Creating & starting a thread for receiving actions
        action_thread = threading.Thread(target=client.receive_actions)
        action_thread.daemon = True
        action_thread.start()
        
        try:
            # Send observations to the server in the main thread
            for i in range(10):
                observation = np.random.randint(0, 10, size=10).astype(np.float32)
                
                if i == 0:
                    state = async_inference_pb2.TRANSFER_BEGIN
                elif i == 9:
                    state = async_inference_pb2.TRANSFER_END
                else:
                    state = async_inference_pb2.TRANSFER_MIDDLE
                
                client.send_observation(observation, state)
                time.sleep(1)


            # Keep the main thread alive to continue receiving actions
            while True:
                time.sleep(1)
                
        except KeyboardInterrupt:
            pass
        
        finally:
            client.stop()

if __name__ == "__main__":
    example_usage()

