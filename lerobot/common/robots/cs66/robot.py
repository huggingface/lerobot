import struct
import socket
import select
from .RobotData import *
import time
from .rtsi import rtsi
    
DEFAULT_TIMEOUT = 10.0

ROBOT_STATE_TYPE = 16
ROBOT_EXCEPTION = 20

def connectETController(ip, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.connect((ip, port))
            return (True, sock)
        except Exception as e:
            sock.close()
            print(f"Failed to connect Robot: {e}, ip: {ip}, port: {port}")
            return (False, None)
        
def disconnectETController(sock):
    if sock:
        try:
            sock.close()
            # print("Disconnected from robot controller")
        except Exception as e:
            print(f"Failed to disconnect: {e}")
        finally:
            sock = None

def rt_connect(ip):
    rt = rtsi(ip)
    rt.connect()
    rt.version_check()
    rt.controller_version()
    rt.output_subscribe
    rt.output_subscribe('actual_digital_output_bits,actual_joint_positions,actual_TCP_pose', 125)
    print("RTSI successfully executed")
    return (True, rt)

    

class RobotController():
    def __init__(self, robot_ip: str = "192.168.101.11"):
        self.robot_ip = robot_ip

    # 连接机器人端口    
    def connect(self):
        self.conSuc_30001, self.sock_30001 = connectETController(f"{self.robot_ip}", 30001)
        self.conSuc_30020, self.sock_30020 = connectETController(f"{self.robot_ip}", 30020)
        self.conSuc_29999, self.sock_29999 = connectETController(f"{self.robot_ip}", 29999)
        self.conSuc_40011, self.sock_40011 = connectETController(f"{self.robot_ip}", 40011)
        self.conSuc_rt,self.rt = rt_connect(f"{self.robot_ip}")
        self.rt.start()
        print(f"Connected to port: 30001{self.conSuc_30001}, 30020:{self.conSuc_30020}, 29999:{self.conSuc_29999}, 40011:{self.conSuc_40011}, rtsi: {self.conSuc_rt}")

    # 断开机器人端口
    def disconnect(self):
        disconnectETController(self.sock_30001)
        disconnectETController(self.sock_30020)
        disconnectETController(self.sock_29999)
        disconnectETController(self.sock_40011)
        self.rt.pause()
        self.rt.disconnect()
        print("Disconnected from all robot ports")

    def interpreter_close(self):
        if self.conSuc_30020 and self.sock_30020 is not None:
            self.sock_30020.sendall(bytes(str("end_interpreter()" + '\n'), "utf-8"))
            time.sleep(0.1)

    def interpreter(self, content):
        if self.conSuc_30020 and self.sock_30020 is not None:
            self.sock_30020.sendall((content + '\n').encode('utf-8'))
            # 调试用打印
            # print("Execute Command: ", content)
            recvData = self.sock_30020.recv(1024)
            return recvData
        
    def sendCMD(self, content):
        if self.conSuc_30001 and self.sock_30001 is not None:
            sendStr = 'def a():\n{}\nend'.format(content)
            try:
                jdata = self.sock_30001.sendall(bytes(sendStr, "utf-8"))
                ret = self.sock_30001.recv(1024)
            except Exception as e:
                return (False, None, None)
            
    def Status_request(self, content):
        if self.conSuc_40011 and self.sock_40011 is not None:
            self.sock_40011.sendall(bytes(str('req 1' + content + '\n'), "utf-8"))
            recvData = self.sock_40011.recv(1024)
            recvData = recvData.decode()
            recvData = recvData.split(":")
            recvData = recvData[1].replace('[', '').replace(']', '').replace(' ', '')
            recvData = recvData.split(",")
            return [float(recvData[0]), float(recvData[1]), float(recvData[2]),
                    float(recvData[3]), float(recvData[4]), float(recvData[5])]
            
    def dashboard_shell(self, content):
        if self.conSuc_29999 and self.sock_29999 is not None:
            self.sock_29999.sendall(bytes(str(content + '\n'), "utf-8"))
            recvData = self.sock_29999.recv(4096)
            recvData = recvData.decode()
            return recvData.replace('\n', '').replace('\r', '')
        
    # def servoJ(self, joint_positions, velocity=0.5, acceleration=0.5, dt=0.002, lookahead_time=0.2, gain=100):
    #     if self.conSuc_30020 and self.sock_30020 is not None:
    #         joint_positions = [round(joint * 180 / 3.1415926, 2) for joint in joint_positions]
    #         sendStr = 'servoJ({}, {}, {}, {}, {}, {})'.format(
    #             joint_positions, velocity, acceleration, dt, lookahead_time, gain)
    #         try:
    #             self.interpreter(sendStr)
    #         except Exception as e:
    #             print(f"Failed to send servoJ command: {e}")