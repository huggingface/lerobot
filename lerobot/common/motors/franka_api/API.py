import grpc
import lerobot.common.motors.franka_api.franka_api_pb2 as franka_api_pb2
import lerobot.common.motors.franka_api.franka_api_pb2_grpc as franka_api_pb2_grpc


class API:
    def __init__(self, server_address):
        self.channel = grpc.insecure_channel(server_address)
        self.stub = franka_api_pb2_grpc.FrankaServiceStub(self.channel)

    def get_JointState(self):
        js = self.stub.GetJointState(franka_api_pb2.Empty())
        return js
    
    def get_JointAngles(self):
        js = self.stub.GetJointState(franka_api_pb2.Empty())
        return js.position
    
    def get_Wrench(self):
        wrench = self.stub.GetWrench(franka_api_pb2.Empty())
        return wrench
    
    def get_CartPose(self):
        cartpose = self.stub.GetEEFPose(franka_api_pb2.Empty())
        return cartpose
    
    def set_JointPosition(self, position):
        response = self.stub.SetJointTarget(franka_api_pb2.JointState(
            name=["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4", "panda_joint5", "panda_joint6", "panda_joint7"],
            position=list(position)
        ))
        return response.message
    
    