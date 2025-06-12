import rospy
import grpc
from concurrent import futures
import time

from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped

import franka_api_pb2
import franka_api_pb2_grpc

class FrankaAPI(franka_api_pb2_grpc.FrankaServiceServicer):
    def __init__(self):
        self.joint_state = None
        self.pose = None
        self.wrench = None

        self.joint_state_sub = rospy.Subscriber("/joint_states", JointState, self.joint_state_cb)
        self.pose_sub = rospy.Subscriber("/cartesian_pose", PoseStamped, self.pose_cb)
        self.wrench_sub = rospy.Subscriber("/force_torque_ext", WrenchStamped, self.wrench_cb)

        self.joint_cmd_pub = rospy.Publisher("/equilibrium_configuration", JointState, queue_size=1)

        # Wait for topics
        timeout = rospy.Duration(5)
        start_time = rospy.Time.now()
        while not rospy.is_shutdown():
            if self.joint_state and self.pose and self.wrench:
                break
            if rospy.Time.now() - start_time > timeout:
                raise RuntimeError("Timeout waiting for ROS topics to publish.")
            rospy.sleep(0.1)

    def joint_state_cb(self, msg):
        self.joint_state = msg

    def pose_cb(self, msg):
        self.pose = msg

    def wrench_cb(self, msg):
        self.wrench = msg

    def GetJointState(self, request, context):
        js = self.joint_state
        return franka_api_pb2.JointState(
            name=list(js.name),
            position=list(js.position),
            velocity=list(js.velocity),
            effort=list(js.effort),
        )

    def GetEEFPose(self, request, context):
        p = self.pose.pose.position
        o = self.pose.pose.orientation
        return franka_api_pb2.Pose(
            x=p.x, y=p.y, z=p.z,
            qx=o.x, qy=o.y, qz=o.z, qw=o.w
        )

    def GetWrench(self, request, context):
        f = self.wrench.wrench.force
        t = self.wrench.wrench.torque
        return franka_api_pb2.Wrench(
            fx=f.x, fy=f.y, fz=f.z,
            tx=t.x, ty=t.y, tz=t.z
        )

    def SetJointTarget(self, request, context):
        try:
            cmd = JointState()
            cmd.name = request.name
            cmd.position = request.position
            rospy.loginfo(f"Publishing joint target: {cmd}")
            self.joint_cmd_pub.publish(cmd)
            return franka_api_pb2.StatusResponse(success=True, message="Command sent")
        except Exception as e:
            return franka_api_pb2.StatusResponse(success=False, message=str(e))

def serve():
    rospy.init_node('franka_api_server')
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    franka_api_pb2_grpc.add_FrankaServiceServicer_to_server(FrankaAPI(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    rospy.loginfo("gRPC server started on port 50051.")
    rospy.spin()

if __name__ == "__main__":
    serve()
