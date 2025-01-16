from concurrent import futures

import grpc

import lerobot.common.api.grpc.sac_pb2 as sac_pb2
import lerobot.common.api.grpc.sac_pb2_grpc as sac_pb2_grpc


class TensorServiceServicer(sac_pb2_grpc.TensorServiceServicer):
    def StreamModelUpdates(self, request, context):
        # Implement logic to stream multiple Tensors
        for _ in range(5):  # example, streaming 5 updates
            yield sac_pb2.ModelState(shape=[2, 3], data=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    def GetModel(self, request, context):
        # Implement logic to return a single Tensor
        return sac_pb2.ModelState(shape=[2, 2], data=[10.0, 20.0, 30.0, 40.0])


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    sac_pb2_grpc.add_TensorServiceServicer_to_server(TensorServiceServicer(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()


if __name__ == "__main__":
    serve()
