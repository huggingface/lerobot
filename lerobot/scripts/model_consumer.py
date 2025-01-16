import grpc

import lerobot.common.api.grpc.sac_pb2 as sac_pb2
import lerobot.common.api.grpc.sac_pb2_grpc as sac_pb2_grpc


def run_client():
    # Create a channel to connect to the server.
    # Use the appropriate server address and port (e.g., 'localhost:50051').
    channel = grpc.insecure_channel("localhost:50051")

    # Create a stub (client) to interact with the TensorService.
    stub = sac_pb2_grpc.TensorServiceStub(channel)

    # 1) Call GetTensor (unary call) and get a single response back.
    request = sac_pb2.ModelRequest(model_name="my_tensor_id")
    response = stub.GetModel(request)
    print("Received single Tensor from GetTensor():")
    print(f"Shape: {response.shape}")
    print(f"Data: {response.data}")

    # 2) Call StreamTensorUpdates (server-streaming call).
    print("Subscribed to StreamTensorUpdates()...")
    stream_request = sac_pb2.ModelRequest(model_name="stream_id")
    for tensor_update in stub.StreamModelUpdates(stream_request):
        # Each `tensor_update` is a Tensor message
        print("Received Tensor update from StreamTensorUpdates():")
        print(f"Shape: {tensor_update.shape}")
        print(f"Data: {tensor_update.data}")


if __name__ == "__main__":
    run_client()
