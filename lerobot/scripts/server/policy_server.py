import torch
import grpc
import time
import threading
import numpy as np
from concurrent import futures

import async_inference_pb2  # type: ignore
import async_inference_pb2_grpc  # type: ignore

from lerobot.common.robot_devices.control_utils import predict_action
from lerobot.common.policies.pretrained import PreTrainedPolicy
from typing import Optional

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PolicyServer(async_inference_pb2_grpc.AsyncInferenceServicer):
    def __init__(self, policy: PreTrainedPolicy = None):        
        self.policy = policy

        # TODO: Add device specification for policy inference

        self.observation = None
        self.clients = []
        self.lock = threading.Lock()
        # keeping a list of all observations received from the robot client
        self.observations = []
    
    def Ready(self, request, context):
        print("Client connected and ready")
        return async_inference_pb2.Empty()
    
    def SendObservations(self, request_iterator, context):
        """Receive observations from the robot client"""
        client_id = context.peer()
        print(f"Receiving observations from {client_id}")
        
        for observation in request_iterator:
            print(
                "Received observation: ",
                f"state={observation.transfer_state}, " 
                f"data size={len(observation.data)} bytes"
                )


            with self.lock:
                self.observation = observation
                self.observations.append(observation)
            
            data = np.frombuffer(self.observation.data, dtype=np.float32)
            print(f"Current observation data: {data}")
            
        return async_inference_pb2.Empty()

    def StreamActions(self, request, context):
        """Stream actions to the robot client"""
        client_id = context.peer()
        print(f"Client {client_id} connected for action streaming")
        
        # Keep track of this client for sending actions
        with self.lock:
            self.clients.append(context)
        
        try:
            # Keep the connection alive
            while context.is_active():
                time.sleep(0.1)
        finally:
            with self.lock:
                if context in self.clients:
                    self.clients.remove(context)
        
        return async_inference_pb2.Empty()

    def _predict_and_queue_action(self, observation):
        """Predict an action based on the observation"""
        # TODO: Implement the logic to predict an action based on the observation
        """
        Ideally, action-prediction should be general and not specific to the policy used.
        That is, this interface should be the same for ACT/VLA/RL-based etc.
        """
        # TODO: Queue the action to be sent to the robot client
        raise NotImplementedError("Not implemented")

    def _generate_and_queue_action(self, observation):
        """Generate an action based on the observation (dummy logic).
        Mainly used for testing purposes"""
        # Just create a random action as a response
        action_data = np.random.rand(50).astype(np.float32).tobytes()
        
        action = async_inference_pb2.Action(
            transfer_state=observation.transfer_state,
            data=action_data
        )
        
        # Send this action to all connected clients
        dead_clients = []
        for client_context in self.clients:
            try:
                if client_context.is_active():
                    client_context.send_initial_metadata([])
                    yield action
                else:
                    dead_clients.append(client_context)
            except:
                dead_clients.append(client_context)
        
        # Clean up dead clients, if any
        for dead in dead_clients:
            if dead in self.clients:
                self.clients.remove(dead)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    async_inference_pb2_grpc.add_AsyncInferenceServicer_to_server(PolicyServer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("PolicyServer started on port 50051")
    
    try:
        while True:
            time.sleep(86400)  # Sleep for a day, or until interrupted
    except KeyboardInterrupt:
        server.stop(0)
        print("Server stopped")

if __name__ == "__main__":
    serve()
