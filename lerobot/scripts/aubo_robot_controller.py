#! /usr/bin/env python
# coding=utf-8
import time
import libpyauboi5
import logging
from logging.handlers import RotatingFileHandler
from multiprocessing import Process, Queue
import os
from math import pi
import socket
import numpy as np

# 创建一个logger
# logger = logging.getLogger()

logger = logging.getLogger('main.robotcontrol')


def logger_init():
    # Log等级总开关
    logger.setLevel(logging.INFO)

    # 创建log目录
    if not os.path.exists('./logfiles'):
        os.mkdir('./logfiles')

    # 创建一个handler，用于写入日志文件
    logfile = './logfiles/robot-ctl-python.log'

    # 以append模式打开日志文件
    # fh = logging.FileHandler(logfile, mode='a')
    fh = RotatingFileHandler(
        logfile, mode='a', maxBytes=1024 * 1024 * 50, backupCount=30)

    # 输出到file的log等级的开关
    fh.setLevel(logging.INFO)

    # 再创建一个handler，用于输出到控制台
    ch = logging.StreamHandler()

    # 输出到console的log等级的开关
    ch.setLevel(logging.INFO)

    # 定义handler的输出格式
    # formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    formatter = logging.Formatter(
        "%(asctime)s [%(thread)u] %(levelname)s: %(message)s")

    # 为文件输出设定格式
    fh.setFormatter(formatter)

    # 控制台输出设定格式
    ch.setFormatter(formatter)

    # 设置文件输出到logger
    logger.addHandler(fh)

    # 设置控制台输出到logger
    logger.addHandler(ch)


class RobotEventType:
    RobotEvent_armCanbusError = 0  # 机械臂CAN总线错误
    RobotEvent_remoteHalt = 1  # 机械臂停止
    RobotEvent_remoteEmergencyStop = 2  # 机械臂远程急停
    RobotEvent_jointError = 3  # 关节错误
    RobotEvent_forceControl = 4  # 力控制
    RobotEvent_exitForceControl = 5  # 退出力控制
    RobotEvent_softEmergency = 6  # 软急停
    RobotEvent_exitSoftEmergency = 7  # 退出软急停
    RobotEvent_collision = 8  # 碰撞
    RobotEvent_collisionStatusChanged = 9  # 碰撞状态改变
    RobotEvent_tcpParametersSucc = 10  # 工具动力学参数设置成功
    RobotEvent_powerChanged = 11  # 机械臂电源开关状态改变
    RobotEvent_ArmPowerOff = 12  # 机械臂电源关闭
    RobotEvent_mountingPoseChanged = 13  # 安装位置发生改变
    RobotEvent_encoderError = 14  # 编码器错误
    RobotEvent_encoderLinesError = 15  # 编码器线数不一致
    RobotEvent_singularityOverspeed = 16  # 奇异点超速
    RobotEvent_currentAlarm = 17  # 机械臂电流异常
    RobotEvent_toolioError = 18  # 机械臂工具端错误
    RobotEvent_robotStartupPhase = 19  # 机械臂启动阶段
    RobotEvent_robotStartupDoneResult = 20  # 机械臂启动完成结果
    RobotEvent_robotShutdownDone = 21  # 机械臂关机结果
    RobotEvent_atTrackTargetPos = 22  # 机械臂轨迹运动到位信号通知
    RobotEvent_SetPowerOnDone = 23  # 设置电源状态完成
    RobotEvent_ReleaseBrakeDone = 24  # 机械臂刹车释放完成
    RobotEvent_robotControllerStateChaned = 25  # 机械臂控制状态改变
    RobotEvent_robotControllerError = 26  # 机械臂控制错误----一般是算法规划出现问题时返回
    RobotEvent_socketDisconnected = 27  # socket断开连接
    RobotEvent_overSpeed = 28  # 超速
    RobotEvent_algorithmException = 29  # 机械臂算法异常
    RobotEvent_boardIoPoweron = 30  # 外部上电信号
    RobotEvent_boardIoRunmode = 31  # 联动/手动
    RobotEvent_boardIoPause = 32  # 外部暂停信号
    RobotEvent_boardIoStop = 33  # 外部停止信号
    RobotEvent_boardIoHalt = 34  # 外部关机信号
    RobotEvent_boardIoEmergency = 35  # 外部急停信号
    RobotEvent_boardIoRelease_alarm = 36  # 外部报警解除信号
    RobotEvent_boardIoOrigin_pose = 37  # 外部回原点信号
    RobotEvent_boardIoAutorun = 38  # 外部自动运行信号
    RobotEvent_safetyIoExternalEmergencyStope = 39  # 外部急停输入01
    RobotEvent_safetyIoExternalSafeguardStope = 40  # 外部保护停止输入02
    RobotEvent_safetyIoReduced_mode = 41  # 缩减模式输入
    RobotEvent_safetyIoSafeguard_reset = 42  # 防护重置
    RobotEvent_safetyIo3PositionSwitch = 43  # 三态开关1
    RobotEvent_safetyIoOperationalMode = 44  # 操作模式
    RobotEvent_safetyIoManualEmergencyStop = 45  # 示教器急停01
    RobotEvent_safetyIoSystemStop = 46  # 系统停止输入
    RobotEvent_alreadySuspended = 47  # 机械臂暂停
    RobotEvent_alreadyStopped = 48  # 机械臂停止
    RobotEvent_alreadyRunning = 49  # 机械臂运行
    RobotEvent_MoveEnterStopState = 1300  # 运动进入到stop阶段
    RobotEvent_None = 999999

    # 非错误事件
    NoError = (RobotEvent_forceControl,
               RobotEvent_exitForceControl,
               RobotEvent_tcpParametersSucc,
               RobotEvent_powerChanged,
               RobotEvent_mountingPoseChanged,
               RobotEvent_robotStartupPhase,
               RobotEvent_robotStartupDoneResult,
               RobotEvent_robotShutdownDone,
               RobotEvent_SetPowerOnDone,
               RobotEvent_ReleaseBrakeDone,
               RobotEvent_atTrackTargetPos,
               RobotEvent_robotControllerStateChaned,
               RobotEvent_robotControllerError,
               RobotEvent_algorithmException,
               RobotEvent_alreadyStopped,
               RobotEvent_alreadyRunning,
               RobotEvent_boardIoPoweron,
               RobotEvent_boardIoRunmode,
               RobotEvent_boardIoPause,
               RobotEvent_boardIoStop,
               RobotEvent_boardIoHalt,
               RobotEvent_boardIoRelease_alarm,
               RobotEvent_boardIoOrigin_pose,
               RobotEvent_boardIoAutorun,
               RobotEvent_safetyIoExternalEmergencyStope,
               RobotEvent_safetyIoExternalSafeguardStope,
               RobotEvent_safetyIoReduced_mode,
               RobotEvent_safetyIoSafeguard_reset,
               RobotEvent_safetyIo3PositionSwitch,
               RobotEvent_safetyIoOperationalMode,
               RobotEvent_safetyIoManualEmergencyStop,
               RobotEvent_safetyIoSystemStop,
               RobotEvent_alreadySuspended,
               RobotEvent_alreadyStopped,
               RobotEvent_alreadyRunning,
               RobotEvent_MoveEnterStopState
               )

    UserPostEvent = (RobotEvent_robotControllerError,
                     RobotEvent_safetyIoExternalSafeguardStope,
                     RobotEvent_safetyIoSystemStop
                     )
    ClearErrorEvent = (RobotEvent_armCanbusError,
                       RobotEvent_remoteEmergencyStop,
                       RobotEvent_jointError,
                       RobotEvent_collision,
                       RobotEvent_collisionStatusChanged,
                       RobotEvent_encoderError,
                       RobotEvent_encoderLinesError,
                       RobotEvent_currentAlarm,
                       RobotEvent_softEmergency,
                       RobotEvent_exitSoftEmergency
                       )

    def __init__(self):
        pass


class RobotErrorType:
    RobotError_SUCC = 0  # 无错误
    RobotError_Base = 2000
    RobotError_RSHD_INIT_FAILED = RobotError_Base + 1  # 库初始化失败
    RobotError_RSHD_UNINIT = RobotError_Base + 2  # 库未初始化
    RobotError_NoLink = RobotError_Base + 3  # 无链接
    RobotError_Move = RobotError_Base + 4  # 机械臂移动错误
    RobotError_ControlError = RobotError_Base + \
                              RobotEventType.RobotEvent_robotControllerError
    RobotError_LOGIN_FAILED = RobotError_Base + 5  # 机械臂登录失败
    RobotError_NotLogin = RobotError_Base + 6  # 机械臂未登录
    RobotError_ERROR_ARGS = RobotError_Base + 7  # 参数错误

    def __init__(self):
        pass


class RobotEvent:
    def __init__(self, event_type=RobotEventType.RobotEvent_None, event_code=0, event_msg=''):
        self.event_type = event_type
        self.event_code = event_code
        self.event_msg = event_msg


# noinspection SpellCheckingInspection
class RobotError(Exception):
    def __init__(self, error_type=RobotErrorType.RobotError_SUCC, error_code=0, error_msg=''):
        self.error_type = error_type
        self.error_cdoe = error_code
        self.error_msg = error_msg

    def __str__(self):
        return "RobotError type{0} code={1} msg={2}".format(self.error_type, self.error_cdoe, self.error_msg)


class RobotDefaultParameters:
    # 缺省的动力学参数
    tool_dynamics = {"position": (0.0, 0.0, 0.0), "payload": 1.0, "inertia": (
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}

    # 缺省碰撞等级
    collision_grade = 6

    def __init__(self):
        pass

    def __str__(self):
        return "Robot Default parameters, tool_dynamics:{0}, collision_grade:{1}".format(self.tool_dynamics,
                                                                                         self.collision_grade)


class RobotMoveTrackType:
    # 圆弧
    ARC_CIR = 2
    # 轨迹
    CARTESIAN_MOVEP = 3
    # 以下四种三阶样条插值曲线都有起始和结束点加速度不连续的情况,不适合与新关节驱动版本
    # 三次样条插值(过控制点),自动优化轨迹运行时间,目前不支持姿态变化
    CARTESIAN_CUBICSPLINE = 4
    # 需要设定三次均匀B样条插值(过控制点)的时间间隔,目前不支持姿态变化
    CARTESIAN_UBSPLINEINTP = 5
    # 三阶样条插值曲线
    JIONT_CUBICSPLINE = 6
    # 可用于轨迹回放
    JOINT_UBSPLINEINTP = 7

    def __init__(self):
        pass


class RobotIOType:
    # 控制柜IO
    ControlBox_DI = 0
    ControlBox_DO = 1
    ControlBox_AI = 2
    ControlBox_AO = 3
    # 用户IO
    User_DI = 4
    User_DO = 5
    User_AI = 6
    User_AO = 7

    def __init__(self):
        pass


class RobotToolIoName:
    tool_io_0 = "T_DI/O_00"
    tool_io_1 = "T_DI/O_01"
    tool_io_2 = "T_DI/O_02"
    tool_io_3 = "T_DI/O_03"

    tool_ai_0 = "T_AI_00"
    tool_ai_1 = "T_AI_01"

    def __init__(self):
        pass


class RobotUserIoName:
    # 控制柜用户ＤＩ
    user_di_00 = "U_DI_00"
    user_di_01 = "U_DI_01"
    user_di_02 = "U_DI_02"
    user_di_03 = "U_DI_03"
    user_di_04 = "U_DI_04"
    user_di_05 = "U_DI_05"
    user_di_06 = "U_DI_06"
    user_di_07 = "U_DI_07"
    user_di_10 = "U_DI_10"
    user_di_11 = "U_DI_11"
    user_di_12 = "U_DI_12"
    user_di_13 = "U_DI_13"
    user_di_14 = "U_DI_14"
    user_di_15 = "U_DI_15"
    user_di_16 = "U_DI_16"
    user_di_17 = "U_DI_17"

    # 控制柜用户ＤＯ
    user_do_00 = "U_DO_00"
    user_do_01 = "U_DO_01"
    user_do_02 = "U_DO_02"
    user_do_03 = "U_DO_03"
    user_do_04 = "U_DO_04"
    user_do_05 = "U_DO_05"
    user_do_06 = "U_DO_06"
    user_do_07 = "U_DO_07"
    user_do_10 = "U_DO_10"
    user_do_11 = "U_DO_11"
    user_do_12 = "U_DO_12"
    user_do_13 = "U_DO_13"
    user_do_14 = "U_DO_14"
    user_do_15 = "U_DO_15"
    user_do_16 = "U_DO_16"
    user_do_17 = "U_DO_17"

    # 控制柜模拟量ＩＯ
    user_ai_00 = "VI0"
    user_ai_01 = "VI1"
    user_ai_02 = "VI2"
    user_ai_03 = "VI3"

    user_ao_00 = "VO0"
    user_ao_01 = "VO1"
    user_ao_02 = "VO2"
    user_ao_03 = "VO3"

    def __init__(self):
        pass


class RobotStatus:
    # 机械臂当前停止
    Stopped = 0
    # 机械臂当前运行
    Running = 1
    # 机械臂当前暂停
    Paused = 2
    # 机械臂当前恢复
    Resumed = 3

    def __init__(self):
        pass


class RobotRunningMode:
    # 机械臂仿真模式
    RobotModeSimulator = 0
    # 机械臂真实模式
    RobotModeReal = 1

    def __init__(self):
        pass


class RobotToolPowerType:
    OUT_0V = 0
    OUT_12V = 1
    OUT_24V = 2

    def __init__(self):
        pass


class RobotToolIoAddr:
    TOOL_DIGITAL_IO_0 = 0
    TOOL_DIGITAL_IO_1 = 1
    TOOL_DIGITAL_IO_2 = 2
    TOOL_DIGITAL_IO_3 = 3

    def __init__(self):
        pass


class RobotCoordType:
    # 基座坐标系
    Robot_Base_Coordinate = 0
    # 末端坐标系
    Robot_End_Coordinate = 1
    # 用户坐标系
    Robot_World_Coordinate = 2

    def __init__(self):
        pass


class RobotCoordCalMethod:
    CoordCalMethod_xOy = 0
    CoordCalMethod_yOz = 1
    CoordCalMethod_zOx = 2
    CoordCalMethod_xOxy = 3
    CoordCalMethod_xOxz = 4
    CoordCalMethod_yOyx = 5
    CoordCalMethod_yOyz = 6
    CoordCalMethod_zOzx = 7
    CoordCalMethod_zOzy = 8

    def __init__(self):
        pass


class RobotToolDigitalIoDir:
    # 输入
    IO_IN = 0
    # 输出
    IO_OUT = 1

    def __init__(self):
        pass


class Auboi5Robot:
    # 客户端个数
    __client_count = 0

    def __init__(self):
        self.rshd = -1
        self.connected = False
        self.last_error = RobotError()
        self.last_event = RobotEvent()
        self.atTrackTargetPos = False
        Auboi5Robot.__client_count += 1

    def __del__(self):
        Auboi5Robot.__client_count -= 1
        self.uninitialize()
        logger.info("client_count={0}".format(Auboi5Robot.__client_count))

    def __str__(self):
        return "RSHD={0}, connected={1}".format(self.rshd, self.connected)

    @staticmethod
    def get_local_time():
        """"
        * FUNCTION:    get_local_time
        * DESCRIPTION: 获取系统当前时间
        * INPUTS:      无输入
        * OUTPUTS:
        * RETURNS:     输出系统当前时间字符串
        * NOTES:
        """
        return time.strftime("%b %d %Y %H:%M:%S", time.localtime(time.time()))

    def robot_event_callback(self, event):
        """"
        * FUNCTION:    robot_event_callback
        * DESCRIPTION: 机械臂事件
        * INPUTS:      无输入
        * OUTPUTS:
        * RETURNS:     系统事件回调函数
        * NOTES:
        """
        print("event={0}".format(event))
        if event['type'] not in RobotEventType.NoError:
            self.last_error = RobotError(
                event['type'], event['code'], event['content'])
        else:
            self.last_event = RobotEvent(
                event['type'], event['code'], event['content'])

        if event['type'] == 34 and event['code'] == 8:
            print("safeguard stop maual release signal caught.")
            # self.move_continue()

    @staticmethod
    def raise_error(error_type, error_code, error_msg):
        """"
        * FUNCTION:    raise_error
        * DESCRIPTION: 抛出异常事件
        * INPUTS:      无输入
        * OUTPUTS:
        * RETURNS:     无
        * NOTES:
        """
        raise RobotError(error_type, error_code, error_msg)

    def check_event(self):
        """"
        * FUNCTION:    check_event
        * DESCRIPTION: 检查机械臂是否发生异常事件
        * INPUTS:      input
        * OUTPUTS:     output
        * RETURNS:     void
        * NOTES:       如果接收到的是异常事件，则函数抛出异常事件
        """
        if self.last_error.error_type != RobotErrorType.RobotError_SUCC:
            raise self.last_error
        if self.rshd == -1 or not self.connected:
            self.raise_error(RobotErrorType.RobotError_NoLink,
                             0, "no socket link")

    @staticmethod
    def initialize():
        """"
        * FUNCTION:    initialize
        * DESCRIPTION: 初始化机械臂控制库
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        result = libpyauboi5.initialize()
        if result == RobotErrorType.RobotError_SUCC:
            return RobotErrorType.RobotError_SUCC
        else:
            return RobotErrorType.RobotError_RSHD_INIT_FAILED

    @staticmethod
    def uninitialize():
        """"
        * FUNCTION:    uninitialize
        * DESCRIPTION: 反初始化机械臂控制库
        * INPUTS:      input
        * OUTPUTS:     output
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        return libpyauboi5.uninitialize()

    def create_context(self):
        """"
        * FUNCTION:    create_context
        * DESCRIPTION: 创建机械臂控制上下文句柄
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RSHD
        * NOTES:
        """
        self.rshd = libpyauboi5.create_context()
        return self.rshd

    def get_context(self):
        """"
        * FUNCTION:    get_context
        * DESCRIPTION: 获取机械臂当前控制上下文
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     上下文句柄RSHD
        * NOTES:
        """
        return self.rshd

    def connect(self, ip='localhost', port=8899):
        """"
        * FUNCTION:    connect
        * DESCRIPTION: 链接机械臂服务器
        * INPUTS:      ip 机械臂服务器地址
        *              port 端口号
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        *
        * NOTES:
        """
        logger.info("ip={0}, port={1}".format(ip, port))
        if self.rshd >= 0:
            if not self.connected:
                if libpyauboi5.login(self.rshd, ip, port) == 0:
                    self.connected = True
                    time.sleep(0.5)
                    return RobotErrorType.RobotError_SUCC
                else:
                    logger.error("login failed!")
                    return RobotErrorType.RobotError_LOGIN_FAILED
            else:
                logger.info("already connected.")
                return RobotErrorType.RobotError_SUCC
        else:
            logger.error("RSHD uninitialized!!!")
            return RobotErrorType.RobotError_RSHD_UNINIT

    def disconnect(self):
        """"
         * FUNCTION:    disconnect
         * DESCRIPTION: 断开机械臂服务器链接
         * INPUTS:
         * OUTPUTS:
         * RETURNS:     成功返回: RobotError.RobotError_SUCC
         *              失败返回: 其他
         * NOTES:
         """
        if self.rshd >= 0 and self.connected:
            libpyauboi5.logout(self.rshd)
            self.connected = False
            time.sleep(0.5)
            return RobotErrorType.RobotError_SUCC
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def robot_startup(self, collision=RobotDefaultParameters.collision_grade,
                      tool_dynamics=RobotDefaultParameters.tool_dynamics):
        """
        * FUNCTION:    robot_startup
        * DESCRIPTION: 启动机械臂
        * INPUTS:      collision：碰撞等级范围(0~10) 缺省：6
        *              tool_dynamics:运动学参数
        *              tool_dynamics = 位置，单位(m) ：{"position": (0.0, 0.0, 0.0),
        *                              负载，单位(kg)： "payload": 1.0,
        *                              惯量：          "inertia": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
        *
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            result = libpyauboi5.robot_startup(
                self.rshd, collision, tool_dynamics)
            time.sleep(0.5)
            return result
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def robot_shutdown(self):
        """
        * FUNCTION:    robot_shutdown
        * DESCRIPTION: 关闭机械臂
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        if self.rshd >= 0 and self.connected:
            result = libpyauboi5.robot_shutdown(self.rshd)
            time.sleep(0.5)
            return result
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def enable_robot_event(self):
        self.check_event()
        if self.rshd >= 0 and self.connected:
            self.set_robot_event_callback(self.robot_event_callback)
            return RobotErrorType.RobotError_SUCC
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def init_profile(self):
        """"
        * FUNCTION:    init_profile
        * DESCRIPTION: 初始化机械臂控制全局属性
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        *
        * NOTES:       调用成功后，系统会自动清理掉之前设置的用户坐标系，
        *              速度，加速度等等属性
        """
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.init_global_move_profile(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def set_joint_maxacc(self, joint_maxacc=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
        """
        * FUNCTION:    set_joint_maxacc
        * DESCRIPTION: 设置六个关节的最大加速度
        * INPUTS:      joint_maxacc:六个关节的最大加速度，单位(rad/s)
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_joint_maxacc(self.rshd, joint_maxacc)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def get_joint_maxacc(self):
        """U_DO_00
        * FUNCTION:    get_joint_maxacc
        * DESCRIPTION: 获取六个关节的最大加速度
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 六个关节的最大加速度单位(rad/s^2)
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_joint_maxacc(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def set_joint_maxvelc(self, joint_maxvelc=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)):
        """
        * FUNCTION:    set_joint_maxvelc
        * DESCRIPTION: 设置六个关节的最大速度
        * INPUTS:      joint_maxvelc:六个关节的最大速度，单位(rad/s)
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_joint_maxvelc(self.rshd, joint_maxvelc)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def get_joint_maxvelc(self):
        """
        * FUNCTION:    get_joint_maxvelc
        * DESCRIPTION: 获取六个关节的最大速度
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 六个关节的最大速度(rad/s)
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_joint_maxvelc(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def set_end_max_line_acc(self, end_maxacc=0.1):
        """
        * FUNCTION:    set_end_max_line_acc
        * DESCRIPTION: 设置机械臂末端最大线加速度
        * INPUTS:      end_maxacc:末端最大加线速度，单位(m/s^2)
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_end_max_line_acc(self.rshd, end_maxacc)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def get_end_max_line_acc(self):
        """
        * FUNCTION:    get_end_max_line_acc
        * DESCRIPTION: 获取机械臂末端最大线加速度
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 机械臂末端最大加速度，单位(m/s^2)
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_end_max_line_acc(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def set_end_max_line_velc(self, end_maxvelc=0.1):
        """
        * FUNCTION:    set_end_max_line_velc
        * DESCRIPTION: 设置机械臂末端最大线速度
        * INPUTS:      end_maxacc:末端最大线速度，单位(m/s)
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_end_max_line_velc(self.rshd, end_maxvelc)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def get_end_max_line_velc(self):
        """
        * FUNCTION:    get_end_max_line_velc
        * DESCRIPTION: 获取机械臂末端最大线速度
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 机械臂末端最大速度，单位(m/s)
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_end_max_line_velc(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def set_end_max_angle_acc(self, end_maxacc=0.1):
        """
        * FUNCTION:    set_end_max_angle_acc
        * DESCRIPTION: 设置机械臂末端最大角加速度
        * INPUTS:      end_maxacc:末端最大加速度，单位(rad/s^2)
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_end_max_angle_acc(self.rshd, end_maxacc)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def get_end_max_angle_acc(self):
        """
        * FUNCTION:    get_end_max_angle_acc
        * DESCRIPTION: 获取机械臂末端最大角加速度
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 机械臂末端最大角加速度，单位(m/s^2)
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_end_max_angle_acc(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def set_end_max_angle_velc(self, end_maxvelc=0.1):
        """
        * FUNCTION:    set_end_max_angle_velc
        * DESCRIPTION: 设置机械臂末端最大角速度
        * INPUTS:      end_maxacc:末端最大速度，单位(rad/s)
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_end_max_line_velc(self.rshd, end_maxvelc)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def get_end_max_angle_velc(self):
        """
        * FUNCTION:    get_end_max_angle_velc
        * DESCRIPTION: 获取机械臂末端最大角速度
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 机械臂末端最大速度，单位(rad/s)
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_end_max_line_velc(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def move_to_target_in_cartesian(self, pos, rpy_xyz):
        """
        * FUNCTION:    move_to_target_in_cartesian
        * DESCRIPTION: 给出笛卡尔坐标值和欧拉角，机械臂轴动到目标位置和姿态
        * INPUTS:      pos:位置坐标（x，y，z），单位(m)
        *              rpy：欧拉角（rx，ry，rz）,单位（度）
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            # 度 -> 弧度
            rpy_xyz = [i / 180.0 * pi for i in rpy_xyz]
            # 欧拉角转四元数
            ori = libpyauboi5.rpy_to_quaternion(self.rshd, rpy_xyz)

            # 逆运算得关节角
            joint_radian = libpyauboi5.get_current_waypoint(self.rshd)

            ik_result = libpyauboi5.inverse_kin(
                self.rshd, joint_radian['joint'], pos, ori)

            logging.info("ik_result====>{0}".format(ik_result))

            # 轴动到目标位置
            result = libpyauboi5.move_joint(self.rshd, ik_result["joint"])
            if result != RobotErrorType.RobotError_SUCC:
                self.raise_error(RobotErrorType.RobotError_Move,
                                 result, "move error")
            else:
                return RobotErrorType.RobotError_SUCC
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def move_joint(self, joint_radian=(0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000), issync=True):
        """
        * FUNCTION:    move_joint
        * DESCRIPTION: 机械臂轴动
        * INPUTS:      joint_radian:六个关节的关节角，单位(rad)
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            result = libpyauboi5.move_joint(self.rshd, joint_radian, issync)
            if result != RobotErrorType.RobotError_SUCC:
                self.raise_error(RobotErrorType.RobotError_Move,
                                 result, "move error")
            else:
                return RobotErrorType.RobotError_SUCC
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def move_line(self, joint_radian=(0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000)):
        """
        * FUNCTION:    move_line
        * DESCRIPTION: 机械臂保持当前姿态直线运动
        * INPUTS:      joint_radian:六个关节的关节角，单位(rad)
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            result = libpyauboi5.move_line(self.rshd, joint_radian)
            if result != RobotErrorType.RobotError_SUCC:
                self.raise_error(RobotErrorType.RobotError_Move,
                                 result, "move error")
            else:
                return RobotErrorType.RobotError_SUCC
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def move_rotate(self, user_coord, rotate_axis, rotate_angle):
        """
        * FUNCTION:    move_rotate
        * DESCRIPTION: 保持当前位置变换姿态做旋转运动
        * INPUTS:      user_coord:用户坐标系
        *              user_coord = {'coord_type': 2,
        *               'calibrate_method': 0,
        *               'calibrate_points':
        *                   {"point1": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        *                    "point2": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        *                    "point3": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        *               'tool_desc':
        *                   {"pos": (0.0, 0.0, 0.0),
        *                    "ori": (1.0, 0.0, 0.0, 0.0)}
        *               }
        *              rotate_axis:转轴(x,y,z) 例如：(1,0,0)表示沿Y轴转动
        *              rotate_angle:旋转角度 单位（rad）
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.move_rotate(self.rshd, user_coord, rotate_axis, rotate_angle)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def clear_offline_track(self):
        """
        * FUNCTION:    clear_offline_track
        * DESCRIPTION: 清理服务器上的非在线轨迹运动数据
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.clear_offline_track(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def append_offline_track_waypoint(self, waypoints):
        """
        * FUNCTION:    append_offline_track_waypoint
        * DESCRIPTION: 向服务器添加非在线轨迹运动路点
        * INPUTS:      waypoints 非在线轨迹运动路点元祖(可包含小于3000个路点), 单位:弧度
        *              例如:((0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        *                    (0.0,-0.000001,-0.000001,0.000001,-0.000001, 0.0))
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.append_offline_track_waypoint(self.rshd, waypoints)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def append_offline_track_file(self, track_file):
        """
        * FUNCTION:    append_offline_track_file
        * DESCRIPTION: 向服务器添加非在线轨迹运动路点文件
        * INPUTS:      路点文件全路径,路点文件的每一行包含六个关节的关节角(弧度),用逗号隔开
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.append_offline_track_file(self.rshd, track_file)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def startup_offline_track(self):
        """
        * FUNCTION:    startup_offline_track
        * DESCRIPTION: 通知服务器启动非在线轨迹运动
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.startup_offline_track(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def stop_offline_track(self):
        """
        * FUNCTION:    stop_offline_track
        * DESCRIPTION: 通知服务器停止非在线轨迹运动
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.stop_offline_track(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def enter_tcp2canbus_mode(self):
        """
        * FUNCTION:    enter_tcp2canbus_mode
        * DESCRIPTION: 通知服务器进入TCP2CANBUS透传模式
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.enter_tcp2canbus_mode(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def leave_tcp2canbus_mode(self):
        """
        * FUNCTION:    leave_tcp2canbus_mode
        * DESCRIPTION: 通知服务器退出TCP2CANBUS透传模式
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.leave_tcp2canbus_mode(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def set_waypoint_to_canbus(self, joint_radian=(0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000)):
        """
        * FUNCTION:    set_waypoint_to_canbus
        * DESCRIPTION: 透传运动路点到CANBUS
        * INPUTS:      joint_radian:六个关节的关节角，单位(rad)
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_waypoint_to_canbus(self.rshd, joint_radian)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def remove_all_waypoint(self):
        """
        * FUNCTION:    remove_all_waypoint
        * DESCRIPTION: 清除所有已经设置的全局路点
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.remove_all_waypoint(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def add_waypoint(self, joint_radian=(0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000)):
        """
        * FUNCTION:    add_waypoint
        * DESCRIPTION: 添加全局路点用于轨迹运动
        * INPUTS:      joint_radian:六个关节的关节角，单位(rad)
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.add_waypoint(self.rshd, joint_radian)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def set_blend_radius(self, blend_radius=0.01):
        """
        * FUNCTION:    set_blend_radius
        * DESCRIPTION: 设置交融半径
        * INPUTS:      blend_radius:交融半径，单位(m)
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            if 0.01 >= blend_radius <= 0.05:
                return libpyauboi5.set_blend_radius(self.rshd, blend_radius)
            else:
                logger.warn("blend radius value range must be 0.01~0.05")
                return RobotErrorType.RobotError_ERROR_ARGS
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def set_circular_loop_times(self, circular_count=1):
        """
        * FUNCTION:    set_circular_loop_times
        * DESCRIPTION: 设置圆运动圈数
        * INPUTS:      circular_count:圆的运动圈数
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        *
        * NOTES:       当circular_count大于0时，机械臂进行圆运动circular_count次
        *              当circular_count等于0时，机械臂进行圆弧轨迹运动
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_circular_loop_times(self.rshd, circular_count)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def set_user_coord(self, user_coord):
        """
        * FUNCTION:    set_user_coord
        * DESCRIPTION: 设置用户坐标系
        * INPUTS:      user_coord:用户坐标系
        *              user_coord = {'coord_type': RobotCoordType.Robot_World_Coordinate,
        *               'calibrate_method': RobotCoordCalMethod.CoordCalMethod_xOy,
        *               'calibrate_points':
        *                   {"point1": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        *                    "point2": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        *                    "point3": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        *               'tool_desc':
        *                   {"pos": (0.0, 0.0, 0.0),
        *                    "ori": (1.0, 0.0, 0.0, 0.0)}
        *               }
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_user_coord(self.rshd, user_coord)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def set_base_coord(self):
        """
        * FUNCTION:    set_base_coord
        * DESCRIPTION: 设置基座坐标系
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_base_coord(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def check_user_coord(self, user_coord):
        """
        * FUNCTION:    check_user_coord
        * DESCRIPTION: 检查用户坐标系参数设置是否合理
        * INPUTS:      user_coord:用户坐标系
        *              user_coord = {'coord_type': 2,
        *               'calibrate_method': 0,
        *               'calibrate_points':
        *                   {"point1": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        *                    "point2": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        *                    "point3": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        *               'tool_desc':
        *                   {"pos": (0.0, 0.0, 0.0),
        *                    "ori": (1.0, 0.0, 0.0, 0.0)}
        *               }
        * OUTPUTS:
        * RETURNS:     合理返回: RobotError.RobotError_SUCC
        *              不合理返回: 其他
        * NOTES:
        """
        return libpyauboi5.check_user_coord(self.rshd, user_coord)

    def set_relative_offset_on_base(self, relative_pos, relative_ori):
        """
        * FUNCTION:    set_relative_offset_on_base
        * DESCRIPTION: 设置基于基座标系运动偏移量
        * INPUTS:      relative_pos=(x, y, z) 相对位移，单位(m)
        *              relative_ori=(w,x,y,z) 相对姿态
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_relative_offset_on_base(self.rshd, relative_pos, relative_ori)

        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def set_relative_offset_on_user(self, relative_pos, relative_ori, user_coord):
        """
        * FUNCTION:    set_relative_offset_on_user
        * DESCRIPTION: 设置基于用户标系运动偏移量
        * INPUTS:      relative_pos=(x, y, z) 相对位移，单位(m)
        *              relative_ori=(w,x,y,z) 目标姿态
        *              user_coord:用户坐标系
        *              user_coord = {'coord_type': 2,
        *               'calibrate_method': 0,
        *               'calibrate_points':
        *                   {"point1": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        *                    "point2": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        *                    "point3": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        *               'tool_desc':
        *                   {"pos": (0.0, 0.0, 0.0),
        *                    "ori": (1.0, 0.0, 0.0, 0.0)}
        *               }
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_relative_offset_on_user(self.rshd, relative_pos, relative_ori, user_coord)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def set_no_arrival_ahead(self):
        """
        * FUNCTION:    set_no_arrival_ahead
        * DESCRIPTION: 取消提前到位设置
        * INPUTS:
        *
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            result = libpyauboi5.set_no_arrival_ahead(self.rshd)
            if result != 0:
                self.raise_error(RobotErrorType.RobotError_Move,
                                 result, "set no arrival ahead error")
            else:
                return RobotErrorType.RobotError_SUCC
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def set_arrival_ahead_distance(self, distance=0.0):
        """
        * FUNCTION:    set_arrival_ahead_distance
        * DESCRIPTION: 设置距离模式下的提前到位距离
        * INPUTS:      distance 提前到位距离 单位（米）
        *
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            result = libpyauboi5.set_arrival_ahead_distance(
                self.rshd, distance)
            if result != 0:
                self.raise_error(RobotErrorType.RobotError_Move,
                                 result, "set arrival ahead distance error")
            else:
                return RobotErrorType.RobotError_SUCC
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def set_arrival_ahead_time(self, sec=0.0):
        """
        * FUNCTION:    set_arrival_ahead_time
        * DESCRIPTION: 设置时间模式下的提前到位时间
        * INPUTS:      sec 提前到位时间　单位（秒）
        *
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            result = libpyauboi5.set_arrival_ahead_time(self.rshd, sec)
            if result != 0:
                self.raise_error(RobotErrorType.RobotError_Move,
                                 result, "set arrival ahead time error")
            else:
                return RobotErrorType.RobotError_SUCC
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def set_arrival_ahead_blend(self, distance=0.0):
        """
        * FUNCTION:    set_arrival_ahead_blend
        * DESCRIPTION: 设置距离模式下交融半径距离
        * INPUTS:      blend 交融半径 单位（米）
        *
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            result = libpyauboi5.set_arrival_ahead_blend(self.rshd, distance)
            if result != 0:
                self.raise_error(RobotErrorType.RobotError_Move,
                                 result, "set arrival ahead blend error")
            else:
                return RobotErrorType.RobotError_SUCC
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def move_track(self, track):
        """
        * FUNCTION:    move_track
        * DESCRIPTION: 轨迹运动
        * INPUTS:      track 轨迹类型，包括如下：
        *              圆弧运动RobotMoveTrackType.ARC_CIR
        *              轨迹运动RobotMoveTrackType.CARTESIAN_MOVEP
        *
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            result = libpyauboi5.move_track(self.rshd, track)
            if result != 0:
                self.raise_error(RobotErrorType.RobotError_Move,
                                 result, "move error")
            else:
                return RobotErrorType.RobotError_SUCC
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def forward_kin(self, joint_radian=(0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000)):
        """
        * FUNCTION:    forward_kin
        * DESCRIPTION: 正解
        * INPUTS:      joint_radian:六个关节的关节角，单位(rad)
        * OUTPUTS:
        * RETURNS:     成功返回: 关节正解结果，结果为详见NOTES
        *              失败返回: None
        *
        * NOTES:       六个关节角 {'joint': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        *              位置 'pos': [-0.06403157614989634, -0.4185973810159096, 0.816883228463401],
        *              姿态 'ori': [-0.11863209307193756, 0.3820514380931854, 0.0, 0.9164950251579285]}
        """
        if self.rshd >= 0:
            return libpyauboi5.forward_kin(self.rshd, joint_radian)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def inverse_kin(self, joint_radian=(0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000),
                    pos=(0.0, 0.0, 0.0), ori=(1.0, 0.0, 0.0, 0.0)):
        """
        * FUNCTION:    forward_kin
        * DESCRIPTION: 逆解
        * INPUTS:      joint_radian:起始点六个关节的关节角，单位(rad)
        *              pos位置(x, y, z)单位(m)
        *              ori位姿(w, x, y, z)
        * OUTPUTS:
        * RETURNS:     成功返回: 关节正解结果，结果为详见NOTES
        *              失败返回: None
        *
        * NOTES:       六个关节角 {'joint': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        *              位置 'pos': [-0.06403157614989634, -0.4185973810159096, 0.816883228463401],
        *              姿态 'ori': [-0.11863209307193756, 0.3820514380931854, 0.0, 0.9164950251579285]}
        """
        if self.rshd >= 0:
            return libpyauboi5.inverse_kin(self.rshd, joint_radian, pos, ori)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def base_to_user(self, pos, ori, user_coord, user_tool):
        """
        * FUNCTION:    base_to_user
        * DESCRIPTION: 用户坐标系转基座坐标系
        * INPUTS:      pos:基座标系下的位置(x, y, z)单位(m)
        *              ori:基座标系下的姿态(w, x, y, z)
        *              user_coord:用户坐标系
        *              user_coord = {'coord_type': 2,
        *               'calibrate_method': 0,
        *               'calibrate_points':
        *                   {"point1": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        *                    "point2": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        *                    "point3": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        *               'tool_desc':
        *                   {"pos": (0.0, 0.0, 0.0),
        *                    "ori": (1.0, 0.0, 0.0, 0.0)}
        *               }
        *               user_tool用户工具描述
        *               user_tool={"pos": (x, y, z), "ori": (w, x, y, z)}
        * OUTPUTS:
        * RETURNS:     成功返回: 返回位置和姿态{"pos": (x, y, z), "ori": (w, x, y, z)}
        *              失败返回: None
        *
        * NOTES:
        """
        return libpyauboi5.base_to_user(self.rshd, pos, ori, user_coord, user_tool)

    def user_to_base(self, pos, ori, user_coord, user_tool):
        """
        * FUNCTION:    user_to_base
        * DESCRIPTION: 用户坐标系转基座标系
        * INPUTS:      pos:用户标系下的位置(x, y, z)单位(m)
        *              ori:用户标系下的姿态(w, x, y, z)
        *              user_coord:用户坐标系
        *              user_coord = {'coord_type': 2,
        *               'calibrate_method': 0,
        *               'calibrate_points':
        *                   {"point1": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        *                    "point2": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        *                    "point3": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
        *               'tool_desc':
        *                   {"pos": (0.0, 0.0, 0.0),
        *                    "ori": (1.0, 0.0, 0.0, 0.0)}
        *               }
        *               user_tool用户工具描述
        *               user_tool={"pos": (x, y, z), "ori": (w, x, y, z)}
        * OUTPUTS:
        * RETURNS:     成功返回: 返回位置和姿态{"pos": (x, y, z), "ori": (w, x, y, z)}
        *              失败返回: None
        *
        * NOTES:
        """
        return libpyauboi5.user_to_base(self.rshd, pos, ori, user_coord, user_tool)

    def base_to_base_additional_tool(self, flange_pos, flange_ori, user_tool):
        """
        * FUNCTION:    base_to_base_additional_tool
        * DESCRIPTION: 基坐标系转基座标得到工具末端点的位置和姿态
        * INPUTS:      pos:基于基座标系的法兰盘中心位置信息(x, y, z)单位(m)
        *              ori:基于基座标系的姿态信息(w, x, y, z)
        *              user_tool用户工具描述
        *              user_tool={"pos": (x, y, z), "ori": (w, x, y, z)}
        * OUTPUTS:
        * RETURNS:     成功返回: 返回基于基座标系的工具末端位置位置和姿态信息{"pos": (x, y, z), "ori": (w, x, y, z)}
        *              失败返回: None
        *
        * NOTES:
        """
        return libpyauboi5.base_to_base_additional_tool(self.rshd, flange_pos, flange_ori, user_tool)

    def rpy_to_quaternion(self, rpy):
        """
        * FUNCTION:    rpy_to_quaternion
        * DESCRIPTION: 欧拉角转四元数
        * INPUTS:      rpy:欧拉角(rx, ry, rz)，单位(m)
        * OUTPUTS:
        * RETURNS:     成功返回: 四元数结果，结果为详见NOTES
        *              失败返回: None
        *
        * NOTES:       四元素(w, x, y, z)
        """
        if self.rshd >= 0:
            return libpyauboi5.rpy_to_quaternion(self.rshd, rpy)
        else:
            logger.warn("RSHD uninitialized !!!")
            return None

    def quaternion_to_rpy(self, ori):
        """
        * FUNCTION:    quaternion_to_rpy
        * DESCRIPTION: 四元数转欧拉角
        * INPUTS:      四元数(w, x, y, z)
        * OUTPUTS:
        * RETURNS:     成功返回: 欧拉角结果，结果为详见NOTES
        *              失败返回: None
        *
        * NOTES:       rpy:欧拉角(rx, ry, rz)，单位(m)
        """
        if self.rshd >= 0:
            return libpyauboi5.quaternion_to_rpy(self.rshd, ori)
        else:
            logger.warn("RSHD uninitialized !!!")
            return None

    def set_tool_end_param(self, tool_end_param):
        """
        * FUNCTION:    set_tool_end_param
        * DESCRIPTION: 设置末端工具参数
        * INPUTS:      末端工具参数： tool_end_param={"pos": (x, y, z), "ori": (w, x, y, z)}
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        *
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_tool_end_param(self.rshd, tool_end_param)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def set_none_tool_dynamics_param(self):
        """
        * FUNCTION:    set_none_tool_dynamics_param
        * DESCRIPTION: 设置无工具的动力学参数
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        *
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_none_tool_dynamics_param(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def set_tool_dynamics_param(self, tool_dynamics):
        """
        * FUNCTION:    set_tool_end_param
        * DESCRIPTION: 设置工具的动力学参数
        * INPUTS:      tool_dynamics:运动学参数
        *              tool_dynamics = 位置，单位(m) ：{"position": (0.0, 0.0, 0.0),
        *                              负载，单位(kg)： "payload": 1.0,
        *                              惯量：          "inertia": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        *
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_tool_dynamics_param(self.rshd, tool_dynamics)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def get_tool_dynamics_param(self):
        """
        * FUNCTION:    get_tool_dynamics_param
        * DESCRIPTION: 获取末端工具参数
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 运动学参数
        *              tool_dynamics = 位置，单位(m) ：{"position": (0.0, 0.0, 0.0),
        *                              负载，单位(kg)： "payload": 1.0,
        *                              惯量：          "inertia": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)}
        *
        *              失败返回: None
        *
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_tool_dynamics_param(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def set_none_tool_kinematics_param(self):
        """
        * FUNCTION:    set_none_tool_kinematics_param
        * DESCRIPTION: 设置无工具运动学参数　
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        *
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_none_tool_kinematics_param(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def set_tool_kinematics_param(self, tool_end_param):
        """
        * FUNCTION:    set_tool_kinematics_param
        * DESCRIPTION: 设置工具的运动学参数　
        * INPUTS:      末端工具参数： tool_end_param={"pos": (x, y, z), "ori": (w, x, y, z)}
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        *
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_tool_kinematics_param(self.rshd, tool_end_param)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def get_tool_kinematics_param(self):
        """
        * FUNCTION:     set_tool_kinematics_param
        * DESCRIPTION: 设置工具的运动学参数　
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 工具的运动学参数
        *               tool_end_param={"pos": (x, y, z), "ori": (w, x, y, z)}
        *
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_tool_kinematics_param(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def move_stop(self):
        """
        * FUNCTION:    move_stop
        * DESCRIPTION: 停止机械臂运动
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.move_stop(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def move_pause(self):
        """
        * FUNCTION:    move_pause
        * DESCRIPTION: 暂停机械臂运动
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.move_pause(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def move_continue(self):
        """
        * FUNCTION:    move_continue
        * DESCRIPTION: 暂停后回复机械臂运动
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.move_continue(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def collision_recover(self):
        """
        * FUNCTION:    collision_recover
        * DESCRIPTION: 机械臂碰撞后恢复
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.collision_recover(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def get_robot_state(self):
        """
        * FUNCTION:    get_robot_state
        * DESCRIPTION: 获取机械臂当前状态
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 机械臂当前状态
        *              机械臂当前停止:RobotStatus.Stopped
        *              机械臂当前运行:RobotStatus.Running
        *              机械臂当前暂停:RobotStatus.Paused
        *              机械臂当前恢复:RobotStatus.Resumed
        *
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_robot_state(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def enter_reduce_mode(self):
        """
        * FUNCTION:    enter_reduce_mode
        * DESCRIPTION: 设置机械臂运动进入缩减模式
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.enter_reduce_mode(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def exit_reduce_mode(self):
        """
        * FUNCTION:    exit_reduce_mode
        * DESCRIPTION: 设置机械臂运动退出缩减模式
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.exit_reduce_mode(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def project_startup(self):
        """
        * FUNCTION:    project_startup
        * DESCRIPTION: 通知机械臂工程启动，服务器同时开始检测安全IO
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.project_startup(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def project_stop(self):
        """
        * FUNCTION:    project_stop
        * DESCRIPTION: 通知机械臂工程停止，服务器停止检测安全IO
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.project_stop(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def set_work_mode(self, mode=0):
        """
        * FUNCTION:    set_work_mode
        * DESCRIPTION: 设置机械臂服务器工作模式
        * INPUTS:      mode:服务器工作模式
        *              机械臂仿真模式:RobotRunningMode.RobotModeSimulator
        *              机械臂真实模式:RobotRunningMode.RobotModeReal
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_work_mode(self.rshd, mode)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_LOGIN_FAILED

    def get_work_mode(self):
        """
        * FUNCTION:    set_work_mode
        * DESCRIPTION: 获取机械臂服务器当前工作模式
        * INPUTS:      mode:服务器工作模式
        *              机械臂仿真模式:RobotRunningMode.RobotModeSimulator
        *              机械臂真实模式:RobotRunningMode.RobotModeReal
        * OUTPUTS:
        * RETURNS:     成功返回: 服务器工作模式
        *              机械臂仿真模式:RobotRunningMode.RobotModeSimulator
        *              机械臂真实模式:RobotRunningMode.RobotModeReal
        *
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_work_mode(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def set_collision_class(self, grade=6):
        """
        * FUNCTION:    set_collision_class
        * DESCRIPTION: 设置机械臂碰撞等级
        * INPUTS:      grade碰撞等级:碰撞等级 范围（0～10）
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_collision_class(self.rshd, grade)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_LOGIN_FAILED

    def is_have_real_robot(self):
        """
        * FUNCTION:    is_have_real_robot
        * DESCRIPTION: 获取当前是否已经链接真实机械臂
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 1：存在 0：不存在
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.is_have_real_robot(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def is_online_mode(self):
        """
        * FUNCTION:    is_online_mode
        * DESCRIPTION: 当前机械臂是否运行在联机模式
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 1：在 0：不在
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.is_online_mode(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def is_online_master_mode(self):
        """
        * FUNCTION:    is_online_master_mode
        * DESCRIPTION: 当前机械臂是否运行在联机主模式
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 1：主模式 0：从模式
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.is_online_master_mode(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def get_joint_status(self):
        """
        * FUNCTION:    get_joint_status
        * DESCRIPTION: 获取机械臂当前状态信息
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 返回六个关节状态，包括：电流，电压，温度
        *              {'joint1': {'current': 电流(毫安), 'voltage': 电压(伏特), 'temperature': 温度(摄氏度)},
        *              'joint2': {'current': 0, 'voltage': 0.0, 'temperature': 0},
        *              'joint3': {'current': 0, 'voltage': 0.0, 'temperature': 0},
        *              'joint4': {'current': 0, 'voltage': 0.0, 'temperature': 0},
        *              'joint5': {'current': 0, 'voltage': 0.0, 'temperature': 0},
        *              'joint6': {'current': 0, 'voltage': 0.0, 'temperature': 0}}
        *
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_joint_status(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def get_current_waypoint(self):
        """
        * FUNCTION:    get_current_waypoint
        * DESCRIPTION: 获取机械臂当前位置信息
        * INPUTS:      grade碰撞等级:碰撞等级 范围（0～10）
        * OUTPUTS:
        * RETURNS:     成功返回: 关节位置信息，结果为详见NOTES
        *              失败返回: None
        *
        * NOTES:       六个关节角 {'joint': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        *              位置 'pos': [-0.06403157614989634, -0.4185973810159096, 0.816883228463401],
        *              姿态 'ori': [-0.11863209307193756, 0.3820514380931854, 0.0, 0.9164950251579285]}
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_current_waypoint(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def get_board_io_config(self, io_type=RobotIOType.User_DO):
        """
        * FUNCTION:    get_board_io_config
        * DESCRIPTION:
        * INPUTS:      io_type：IO类型：RobotIOType
        * OUTPUTS:
        * RETURNS:     成功返回: IO配置
        *               [{"id": ID
        *                 "name": "IO名字"
        *                 "addr": IO地址
        *                 "type": IO类型
        *                 "value": IO当前值},]
        *
        *              失败返回: None
        * NOTES:       RobotIOType详见class RobotIOType
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_board_io_config(self.rshd, io_type)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def get_board_io_status(self, io_type, io_name):
        """
        * FUNCTION:    get_board_io_status
        * DESCRIPTION: 获取IO状态
        * INPUTS:      io_type:类型
        *              io_name:名称 RobotUserIoName.user_dx_xx
        * OUTPUTS:
        * RETURNS:     成功返回: IO状态 double数值(数字IO，返回0或1,模拟IO返回浮点数）
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_board_io_status(self.rshd, io_type, io_name)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def set_board_io_status(self, io_type, io_name, io_value):
        """
        * FUNCTION:    set_board_io_status
        * DESCRIPTION: 设置IO状态
        * INPUTS:      io_type:类型
        *              io_name:名称 RobotUserIoName.user_dx_xx
        *              io_value:状态数值(数字IO，返回0或1,模拟IO返回浮点数）
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        # self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_board_io_status(self.rshd, io_type, io_name, io_value)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_LOGIN_FAILED

    def set_tool_power_type(self, power_type=RobotToolPowerType.OUT_0V):
        """
        * FUNCTION:    set_tool_power_type
        * DESCRIPTION: 设置工具端电源类型
        * INPUTS:      power_type:电源类型
        *              RobotToolPowerType.OUT_0V
        *              RobotToolPowerType.OUT_12V
        *              RobotToolPowerType.OUT_24V
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_tool_power_type(self.rshd, power_type)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_LOGIN_FAILED

    def get_tool_power_type(self):
        """
        * FUNCTION:    get_tool_power_type
        * DESCRIPTION: 获取工具端电源类型
        * INPUTS:      power_type:电源类型

        * OUTPUTS:
        * RETURNS:     成功返回: 电源类型，包括如下：
        *                       RobotToolPowerType.OUT_0V
        *                       RobotToolPowerType.OUT_12V
        *                       RobotToolPowerType.OUT_24V
        *
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_tool_power_type(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def set_tool_io_type(self, io_addr=RobotToolIoAddr.TOOL_DIGITAL_IO_0,
                         io_type=RobotToolDigitalIoDir.IO_OUT):
        """
        * FUNCTION:    set_tool_io_type
        * DESCRIPTION: 设置工具端数字IO类型
        * INPUTS:      io_addr:工具端IO地址 详见class RobotToolIoAddr
        *              io_type:工具端IO类型 详见class RobotToolDigitalIoDir

        * OUTPUTS:
        * RETURNS:     成功返回: IO类型，包括如下：
        *                       RobotToolDigitalIoDir.IO_IN
        *                       RobotToolDigitalIoDir.IO_OUT
        *
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_tool_io_type(self.rshd, io_addr, io_type)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_LOGIN_FAILED

    def get_tool_power_voltage(self):
        """
        * FUNCTION:    get_tool_power_voltage
        * DESCRIPTION: 获取工具端电压数值
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 返回电压数值，单位（伏特）
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_tool_power_voltage(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def get_tool_io_status(self, io_name):
        """
        * FUNCTION:    get_tool_io_status
        * DESCRIPTION: 获取工具端IO状态
        * INPUTS:      io_name:IO名称

        * OUTPUTS:
        * RETURNS:     成功返回: 返回工具端IO状态
        *
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_tool_io_status(self.rshd, io_name)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def set_tool_io_status(self, io_name, io_status):
        """
        * FUNCTION:    set_tool_io_status
        * DESCRIPTION: 设置工具端IO状态
        * INPUTS:      io_name：工具端IO名称
        *              io_status:工具端IO状态: 取值范围（0或1）

        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_tool_do_status(self.rshd, io_name, io_status)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_LOGIN_FAILED

    def startup_excit_traj_track(self, track_file='', track_type=0, subtype=0):
        """
        * FUNCTION:    startup_excit_traj_track
        * DESCRIPTION: 通知服务器启动辨识轨迹运动
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.startup_excit_traj_track(self.rshd, track_file, track_type, subtype)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_NotLogin

    def get_dynidentify_results(self):
        """
        * FUNCTION:    get_dynidentify_results
        * DESCRIPTION: 获取辨识结果
        * INPUTS:
        * OUTPUTS:
        * RETURNS:     成功返回: 辨识结果数组
        *              失败返回: None
        * NOTES:
        """
        self.check_event()
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.get_dynidentify_results(self.rshd)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return None

    def set_robot_event_callback(self, callback):
        """
        * FUNCTION:    set_robot_event_callback
        * DESCRIPTION: 设置机械臂事件回调函数
        * INPUTS:      callback：回调函数名称
        * OUTPUTS:
        * RETURNS:     成功返回: RobotError.RobotError_SUCC
        *              失败返回: 其他
        * NOTES:
        """
        if self.rshd >= 0 and self.connected:
            return libpyauboi5.set_robot_event_callback(self.rshd, callback)
        else:
            logger.warn("RSHD uninitialized or not login!!!")
            return RobotErrorType.RobotError_LOGIN_FAILED


class RobotControllerDataReader:
    # TCP 端口订阅pose和joint

    def __init__(self, ip='172.31.0.199', port=8891):
        self.robot_ip = ip
        self.robot_port = port
        self.sock = None
        self.buffer = ""

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.sock.connect((self.robot_ip, self.robot_port))
            print(f"TCP成功连接到机器人 {self.robot_ip}:{self.robot_port}")
        except Exception as e:
            print(f"连接失败: {e}")
            return False
        return True

    def send_command(self, command="GET_TCP_POSITION\n"):
        if self.sock:
            try:
                self.sock.sendall(command.encode())
            except Exception as e:
                return False
        return True

    def parse_robot_data(self, data):
        try:
            data_str = data.decode('utf-8')
            if "<PACK_BEGIN" in data_str:
                self.buffer += data_str
                if "PACK_END>" in self.buffer:
                    packet = self.buffer
                    self.buffer = ""
                    joint_pos_start = packet.find('"jointPos":[')
                    if joint_pos_start != -1:
                        joint_pos_end = packet.find(']', joint_pos_start)
                        joint_pos_str = packet[joint_pos_start + len('"jointPos":['):joint_pos_end]
                        joint_pos_list = [float(x) for x in joint_pos_str.split(',')]
                        joint_positions = np.array(joint_pos_list)
                        return joint_positions
                    else:
                        print("robot获取数据tcp 数据中未找到 'jointPos'")
                        return None
        except Exception as e:
            print(f"解析数据时出错: {e}")
            return None

    def receive_data(self):
        if not self.sock:
            print("robot获取数据tcp 未连接到机器人")
            return None

        try:
            while True:
                data = self.sock.recv(4096)
                if data:
                    joint_positions = self.parse_robot_data(data)
                    if joint_positions is not None:
                        return joint_positions
                else:
                    print("robot获取数据tcp 未收到数据，可能连接已关闭")
                    break
        except Exception as e:
            print(f"接收数据时出错: {e}")
        finally:
            self.close_connection()

    def close_connection(self):
        if self.sock:
            try:
                self.sock.close()
                print("robot获取数据tcp连接已关闭")
            except Exception as e:
                print(f"robot获取数据tcp关闭连接时出错: {e}")
        self.sock = None

    def get_joint_position(self):
        if self.connect():
            if self.send_command("GET_TCP_POSITION\n"):
                joint_positions = self.receive_data()
                if joint_positions is not None:
                    return joint_positions
        return None


# 测试函数
def test(test_count):
    # 初始化logger
    logger_init()

    # 启动测试
    logger.info("{0} test beginning...".format(Auboi5Robot.get_local_time()))

    # 系统初始化
    Auboi5Robot.initialize()

    # 创建机械臂控制类
    robot = Auboi5Robot()

    # 创建上下文
    handle = robot.create_context()

    # 打印上下文
    logger.info("robot.rshd={0}".format(handle))

    try:

        # 链接服务器
        ip = 'localhost'
        # ip = '192.168.199.200'

        port = 8899
        result = robot.connect(ip, port)

        if result != RobotErrorType.RobotError_SUCC:
            logger.info("connect server{0}:{1} failed.".format(ip, port))
        else:
            # # 重新上电
            # robot.robot_shutdown()
            #
            # # 上电
            robot.robot_startup()
            #
            # # 设置碰撞等级
            robot.set_collision_class(7)

            # 设置工具端电源为１２ｖ
            # robot.set_tool_power_type(RobotToolPowerType.OUT_12V)

            # 设置工具端ＩＯ_0为输出
            # robot.set_tool_io_type(RobotToolIoAddr.TOOL_DIGITAL_IO_0, RobotToolDigitalIoDir.IO_OUT)

            # 获取工具端ＩＯ_0当前状态
            # tool_io_status = robot.get_tool_io_status(RobotToolIoName.tool_io_0)
            # logger.info("tool_io_0={0}".format(tool_io_status))

            # 设置工具端ＩＯ_0状态
            # robot.set_tool_io_status(RobotToolIoName.tool_io_0, 1)

            # 获取控制柜用户DO
            # io_config = robot.get_board_io_config(RobotIOType.User_DO)

            # 输出DO配置
            # logger.info(io_config)

            # 当前机械臂是否运行在联机模式
            # logger.info("robot online mode is {0}".format(robot.is_online_mode()))

            # 循环测试
            while test_count > 0:
                test_count -= 1

                joint_status = robot.get_joint_status()
                logger.info("joint_status={0}".format(joint_status))

                # 初始化全局配置文件
                robot.init_profile()

                # 设置关节最大加速度
                robot.set_joint_maxacc((1.5, 1.5, 1.5, 1.5, 1.5, 1.5))

                # 设置关节最大加速度
                robot.set_joint_maxvelc((1.5, 1.5, 1.5, 1.5, 1.5, 1.5))

                joint_radian = (0.541678, 0.225068, -0.948709,
                                0.397018, -1.570800, 0.541673)
                logger.info("move joint to {0}".format(joint_radian))

                robot.move_joint(joint_radian)

                # 获取关节最大加速度
                logger.info(robot.get_joint_maxacc())

                # 正解测试
                fk_ret = robot.forward_kin(
                    (-0.000003, -0.127267, -1.321122, 0.376934, -1.570796, -0.000008))
                logger.info(fk_ret)

                # 逆解
                joint_radian = (0.000000, 0.000000, 0.000000,
                                0.000000, 0.000000, 0.000000)
                ik_result = robot.inverse_kin(
                    joint_radian, fk_ret['pos'], fk_ret['ori'])
                logger.info(ik_result)

                # 轴动1
                joint_radian = (0.000000, 0.000000, 0.000000,
                                0.000000, 0.000000, 0.000000)
                logger.info("move joint to {0}".format(joint_radian))
                robot.move_joint(joint_radian)

                # 轴动2
                joint_radian = (0.541678, 0.225068, -0.948709,
                                0.397018, -1.570800, 0.541673)
                logger.info("move joint to {0}".format(joint_radian))
                robot.move_joint(joint_radian)

                # 轴动3
                joint_radian = (-0.000003, -0.127267, -1.321122,
                                0.376934, -1.570796, -0.000008)
                logger.info("move joint to {0}".format(joint_radian))
                robot.move_joint(joint_radian)

                # 设置机械臂末端最大线加速度(m/s)
                robot.set_end_max_line_acc(0.5)

                # 获取机械臂末端最大线加速度(m/s)
                robot.set_end_max_line_velc(0.2)

                # 清除所有已经设置的全局路点
                robot.remove_all_waypoint()

                # 添加全局路点1,用于轨迹运动
                joint_radian = (-0.000003, -0.127267, -1.321122,
                                0.376934, -1.570796, -0.000008)
                robot.add_waypoint(joint_radian)

                # 添加全局路点2,用于轨迹运动
                joint_radian = (-0.211675, -0.325189, -1.466753,
                                0.429232, -1.570794, -0.211680)
                robot.add_waypoint(joint_radian)

                # 添加全局路点3,用于轨迹运动
                joint_radian = (-0.037186, -0.224307, -1.398285,
                                0.396819, -1.570796, -0.037191)
                robot.add_waypoint(joint_radian)

                # 设置圆运动圈数
                robot.set_circular_loop_times(3)

                # 圆弧运动
                logger.info("move_track ARC_CIR")
                robot.move_track(RobotMoveTrackType.ARC_CIR)

                # 清除所有已经设置的全局路点
                robot.remove_all_waypoint()

                # 机械臂轴动 回到0位
                joint_radian = (0.000000, 0.000000, 0.000000,
                                0.000000, 0.000000, 0.000000)
                logger.info("move joint to {0}".format(joint_radian))
                robot.move_joint(joint_radian)

            # 断开服务器链接
            robot.disconnect()

    except RobotError as e:
        logger.error("{0} robot Event:{1}".format(robot.get_local_time(), e))

    finally:
        # 断开服务器链接
        if robot.connected:
            # 关闭机械臂
            robot.robot_shutdown()
            # 断开机械臂链接
            robot.disconnect()
        # 释放库资源
        Auboi5Robot.uninitialize()
        logger.info("{0} test completed.".format(Auboi5Robot.get_local_time()))


def step_test():
    # 初始化logger
    logger_init()

    # 启动测试
    logger.info("{0} test beginning...".format(Auboi5Robot.get_local_time()))

    # 系统初始化
    Auboi5Robot.initialize()

    # 创建机械臂控制类
    robot = Auboi5Robot()

    # 创建上下文
    handle = robot.create_context()

    # 打印上下文
    logger.info("robot.rshd={0}".format(handle))

    try:

        # 链接服务器
        ip = 'localhost'
        port = 8899
        result = robot.connect(ip, port)

        if result != RobotErrorType.RobotError_SUCC:
            logger.info("connect server{0}:{1} failed.".format(ip, port))
        else:
            # 重新上电
            robot.robot_shutdown()

            # 上电
            robot.robot_startup()

            # 设置碰撞等级
            robot.set_collision_class(7)

            # # 初始化全局配置文件
            # robot.init_profile()
            #
            # # logger.info(robot.get_board_io_config(RobotIOType.User_DI))
            #
            # # 获取当前位置
            # logger.info(robot.get_current_waypoint())
            #
            # joint_radian = (0, 0, 0, 0, 0, 0)
            # # 轴动到初始位置
            # robot.move_joint(joint_radian)
            #
            # # 沿Ｚ轴运动0.1毫米
            # current_pos = robot.get_current_waypoint()
            #
            # current_pos['pos'][2] -= 0.001
            #
            # ik_result = robot.inverse_kin(current_pos['joint'], current_pos['pos'], current_pos['ori'])
            # logger.info(ik_result)
            #
            # # joint_radian = (0.541678, 0.225068, -0.948709, 0.397018, -1.570800, 0.541673)
            # # logger.info("move joint to {0}".format(joint_radian))
            # # robot.move_joint(joint_radian)
            #
            # robot.move_line(ik_result['joint'])

            # 断开服务器链接
            robot.disconnect()

    except RobotError as e:
        logger.error("robot Event:{0}".format(e))

    finally:
        # 断开服务器链接
        if robot.connected:
            # 断开机械臂链接
            robot.disconnect()
        # 释放库资源
        Auboi5Robot.uninitialize()
        logger.info("{0} test completed.".format(Auboi5Robot.get_local_time()))


def excit_traj_track_test():
    # 初始化logger
    logger_init()

    # 启动测试
    logger.info("{0} test beginning...".format(Auboi5Robot.get_local_time()))

    # 系统初始化
    Auboi5Robot.initialize()

    # 创建机械臂控制类
    robot = Auboi5Robot()

    # 创建上下文
    handle = robot.create_context()

    # 打印上下文
    logger.info("robot.rshd={0}".format(handle))

    try:

        # 链接服务器
        ip = 'localhost'
        port = 8899
        result = robot.connect(ip, port)

        if result != RobotErrorType.RobotError_SUCC:
            logger.info("connect server{0}:{1} failed.".format(ip, port))
        else:

            # 重新上电
            # robot.robot_shutdown()

            # 上电
            # robot.robot_startup()

            # 设置碰撞等级
            # robot.set_collision_class(7)

            joint_radian = (0, 0, 0, 0, 0, 0)
            # 轴动到初始位置
            robot.move_joint(joint_radian)

            logger.info("starup excit traj track....")

            # 启动辨识轨迹
            # robot.startup_excit_traj_track("dynamics_exciting_trajectories/excitTraj1.offt", 1, 0)

            # 延时两秒等待辨识结果
            # time.sleep(5)

            # 获取辨识结果
            dynidentify_ret = robot.get_dynidentify_results()
            logger.info("dynidentify result={0}".format(dynidentify_ret))
            for i in range(0, 54):
                dynidentify_ret[i] = dynidentify_ret[i] / 1024.0
            logger.info("dynidentify result={0}".format(dynidentify_ret))

            # 断开服务器链接
            robot.disconnect()

    except RobotError as e:
        logger.error("robot Event:{0}".format(e))

    finally:
        # 断开服务器链接
        if robot.connected:
            # 断开机械臂链接
            robot.disconnect()
        # 释放库资源
        Auboi5Robot.uninitialize()


def move_rotate_test():
    # 初始化logger
    logger_init()

    # 启动测试                # 初始化全局配置文件
    robot.init_profile()

    # 设置关节最大加速度
    robot.set_joint_maxacc((1.5, 1.5, 1.5, 1.5, 1.5, 1.5))

    # 设置关节最大加速度
    robot.set_joint_maxvelc((1.5, 1.5, 1.5, 1.5, 1.5, 1.5))
    # 系统初始化
    Auboi5Robot.initialize()

    # 创建机械臂控制类
    robot = Auboi5Robot()

    # 创建上下文
    handle = robot.create_context()

    # 打印上下文
    logger.info("robot.rshd={0}".format(handle))

    try:

        # 链接服务器
        ip = 'localhost'
        port = 8899
        result = robot.connect(ip, port)

        if result != RobotErrorType.RobotError_SUCC:
            logger.info("connect server{0}:{1} failed.".format(ip, port))
        else:

            # 重新上电
            # robot.robot_shutdown()

            # 上电
            # robot.robot_startup()

            # 设置碰撞等级
            # robot.set_collision_class(7)

            # joint_radian = (1, 0, 0, 0, 0, 0)
            # # 轴动到初始位置
            # robot.move_joint(joint_radian)

            joint_radian = (0.541678, 0.225068, -0.948709,
                            0.397018, -1.570800, 0.541673)
            logger.info("move joint to {0}".format(joint_radian))
            robot.move_joint(joint_radian)

            # 获取当前位置
            current_pos = robot.get_current_waypoint()

            # 工具转轴的向量（相对于法兰盘，这样需要测量得到x,y,z本测试样例默认以x=0,y=0,ｚ轴为0.1米）
            tool_pos_on_end = (0, 0, 0.10)

            # 工具姿态（w,x,y,z 相对于法兰盘，不知道的情况下，默认填写如下信息）
            tool_ori_on_end = (1, 0, 0, 0)

            tool_desc = {"pos": tool_pos_on_end, "ori": tool_ori_on_end}

            # 得到法兰盘工具末端点相对于基座坐标系中的位置
            tool_pos_on_base = robot.base_to_base_additional_tool(current_pos['pos'],
                                                                  current_pos['ori'],
                                                                  tool_desc)

            logger.info("current_pos={0}".format(current_pos['pos'][0]))

            logger.info("tool_pos_on_base={0}".format(
                tool_pos_on_base['pos'][0]))

            # 讲工具转轴向量平移到基座坐标系下(旋转方向符合右手准则)
            rotate_axis = map(lambda a, b: a - b,
                              tool_pos_on_base['pos'], current_pos['pos'])

            logger.info("rotate_axis={0}".format(rotate_axis))

            # 坐标系默认使用基座坐标系（默认填写下面的值就可以了）
            user_coord = {'coord_type': RobotCoordType.Robot_Base_Coordinate,
                          'calibrate_method': 0,
                          'calibrate_points':
                              {"point1": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                               "point2": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                               "point3": (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)},
                          'tool_desc':
                              {"pos": (0.0, 0.0, 0.0),
                               "ori": (1.0, 0.0, 0.0, 0.0)}
                          }

            # 调用转轴旋转接口，最后一个参数为旋转角度（弧度）
            robot.move_rotate(user_coord, rotate_axis, 1)

            # 断开服务器链接
            robot.disconnect()

    except RobotError as e:
        logger.error("robot Event:{0}".format(e))

    finally:
        # 断开服务器链接
        if robot.connected:
            # 断开机械臂链接
            robot.disconnect()
        # 释放库资源
        Auboi5Robot.uninitialize()


class robot_safeguard_stop_manual_release_event_proc(Process):
    def __init__(self):
        Process.__init__(self)

    def run(self):
        # 初始化logger
        logger_init()

        # 启动测试
        logger.info("{0} robot_safeguard_stop_manual_release_event_proc starup...".format(
            Auboi5Robot.get_local_time()))

        # # 系统初始化
        # Auboi5Robot.initialize()

        # 创建机械臂控制类
        robot = Auboi5Robot()

        # 创建上下文
        handle = robot.create_context()

        # 打印上下文
        logger.info("robot.rshd={0}".format(handle))

        try:
            # 链接服务器
            # ip = 'localhost'
            ip = '192.168.88.57'
            port = 8899
            result = robot.connect(ip, port)

            robot.enable_robot_event()

            if result != RobotErrorType.RobotError_SUCC:
                logger.info("connect server{0}:{1} failed.".format(ip, port))
            else:
                while True:
                    time.sleep(2)
                    print("-------------------------", robot)

            # 断开服务器链接
            robot.disconnect()

        except RobotError as e:
            logger.error("robot Event:{0}".format(e))

        except KeyboardInterrupt:
            # 断开服务器链接
            if robot.connected:
                # 断开机械臂链接
                robot.disconnect()
            # 释放库资源
            Auboi5Robot.uninitialize()


def test_rsm():
    # 初始化logger
    logger_init()

    # 启动测试
    logger.info("{0} test beginning...".format(Auboi5Robot.get_local_time()))

    # 系统初始化
    Auboi5Robot.initialize()

    # 创建机械臂控制类
    robot = Auboi5Robot()

    # 创建上下文
    handle = robot.create_context()

    # 打印上下文
    logger.info("robot.rshd={0}".format(handle))

    try:

        # 链接服务器
        # ip = 'localhost'
        ip = '172.31.0.199'
        port = 8899
        result = robot.connect(ip, port)

        # robot.enable_robot_event()

        if result != RobotErrorType.RobotError_SUCC:
            logger.info("connect server{0}:{1} failed.".format(ip, port))
        else:

            # proc = robot_safeguard_stop_manual_release_event_proc()
            # proc.daemon = True
            # proc.start()
            # time.sleep(0.2)

            robot.project_stop()

            robot.project_startup()

            # robot.move_pause()

            # joint_radian = (0, 0, 0, 0, 0, 0)
            # 轴动到初始位置
            # robot.move_joint(joint_radian)

            # 初始化全局配置文件
            robot.init_profile()

            # 设置关节最大加速度
            robot.set_joint_maxacc((2, 2, 2, 2, 2, 2))

            # 设置关节最大加速度
            robot.set_joint_maxvelc((2, 2, 2, 2, 2, 2))

            while True:
                waypoint = robot.get_current_waypoint()
                print("waypoint: ", waypoint)
                print("----------------------------------------------")

            # time.sleep(0.05)

            # rel = robot.set_board_io_status(RobotIOType.User_DO, RobotUserIoName.user_do_02, 0)
            # print(rel)
            # print("++++++++++++++++++++++++")
            # result = robot.get_board_io_status(RobotIOType.User_DO, RobotUserIoName.user_do_02)
            # print(result)
            # print("*********************************")

            # print("++++++++++++++++++++++++", robot)
            # joint_radian = (0.541678, 0.225068, -0.948709,
            #                  0.397018, -1.570800, 0.541673)
            # fk = robot.forward_kin(joint_radian)

            # print(fk['pos'])
            # print(fk['ori'])

            # joint_radian = (0.00, 0.00, 0.00, 0.00, 0.00, 0.00)

            # ik = robot.inverse_kin(joint_radian, fk['pos'], fk['ori'])

            # print(ik["joint"])
            # rel1 = robot.set_board_io_status(RobotIOType.User_DO, RobotUserIoName.user_do_02, 0)
            # print(rel1)
            # print("++++++++++++++++++++++++")

            robot.project_stop()

            # 断开服务器链接
            robot.disconnect()

    except RobotError as e:
        logger.error("robot Event:{0}".format(e))

    finally:
        # 断开服务器链接
        if robot.connected:
            # 断开机械臂链接
            robot.disconnect()
        # 释放库资源
        Auboi5Robot.uninitialize()


class GetRobotWaypointProcess(Process):
    def __init__(self):
        Process.__init__(self)
        self.isRunWaypoint = False
        self._waypoints = None

    def startMoveList(self, waypoints):
        if self.isRunWaypoint == True:
            return False
        else:
            self._waypoints = waypoints

    def run(self):
        # 初始化logger
        logger_init()

        # 启动测试
        logger.info("{0} test beginning...".format(
            Auboi5Robot.get_local_time()))

        # 系统初始化
        Auboi5Robot.initialize()

        # 创建机械臂控制类
        robot = Auboi5Robot()

        # 创建上下文
        handle = robot.create_context()

        # 打印上下文
        logger.info("robot.rshd={0}".format(handle))

        try:
            # 链接服务器
            # ip = 'localhost'
            ip = '192.168.65.131'
            port = 8899
            result = robot.connect(ip, port)

            if result != RobotErrorType.RobotError_SUCC:
                logger.info("connect server{0}:{1} failed.".format(ip, port))
            else:
                while True:
                    time.sleep(2)
                    waypoint = robot.get_current_waypoint()
                    print(waypoint)
                    print("----------------------------------------------")

                    # 断开服务器链接
                robot.disconnect()

        except RobotError as e:
            logger.error("robot Event:{0}".format(e))

        except KeyboardInterrupt:
            # 断开服务器链接
            if robot.connected:
                # 断开机械臂链接
                robot.disconnect()
            # 释放库资源
            Auboi5Robot.uninitialize()
            print("get  waypoint run end-------------------------")


def runWaypoint(queue):
    while True:
        # while not queue.empty():
        print(queue.get(True))


def test_process_demo():
    # 初始化logger
    logger_init()

    # 启动测试
    logger.info("{0} test beginning...".format(Auboi5Robot.get_local_time()))

    # 系统初始化
    Auboi5Robot.initialize()

    # 创建机械臂控制类
    robot = Auboi5Robot()

    # 创建上下文
    handle = robot.create_context()

    # 打印上下文
    logger.info("robot.rshd={0}".format(handle))

    try:

        # time.sleep(0.2)
        process_get_robot_current_status = GetRobotWaypointProcess()
        # process_get_robot_current_status.daemon = True
        # process_get_robot_current_status.start()
        # time.sleep(0.2)

        queue = Queue()

        p = Process(target=runWaypoint, args=(queue,))
        p.start()
        time.sleep(5)
        print("process started.")

        # 链接服务器
        # ip = 'localhost'
        ip = '172.31.0.199'
        port = 8899
        result = robot.connect(ip, port)

        if result != RobotErrorType.RobotError_SUCC:
            logger.info("connect server{0}:{1} failed.".format(ip, port))
        else:
            robot.project_startup()
            robot.enable_robot_event()
            robot.init_profile()
            joint_maxvelc = (2.596177, 2.596177, 2.596177,
                             3.110177, 3.110177, 3.110177)
            joint_maxacc = (17.308779 / 2.5, 17.308779 / 2.5, 17.308779 /
                            2.5, 17.308779 / 2.5, 17.308779 / 2.5, 17.308779 / 2.5)
            robot.set_joint_maxacc(joint_maxacc)
            robot.set_joint_maxvelc(joint_maxvelc)
            robot.set_arrival_ahead_blend(0.05)
            while True:
                time.sleep(1)

                joint_radian = (0.541678, 0.225068, -0.948709,
                                0.397018, -1.570800, 0.541673)
                fk = robot.forward_kin(joint_radian)

                printf(fk['pos'])
                print(fk['ori'])

                joint_radian = (0.00, 0.00, 0.00, 0.00, 0.00, 0.00)

                ik = robot.inverse_kin(joint_radian, fk['pos'], fk['ori'])

                print(ik["joint"])
                # joint_radian = (55.5/180.0*pi, -20.5/180.0*pi, -72.5 /
                #                 180.0*pi, 38.5/180.0*pi, -90.5/180.0*pi, 55.5/180.0*pi)
                # robot.move_joint(joint_radian, True)

                # joint_radian = (0, 0, 0, 0, 0, 0)
                # robot.move_joint(joint_radian, True)

                print("-----------------------------")

                # queue.put(joint_radian)

                robot.project_stop()

                # time.sleep(5)

                # process_get_robot_current_status.test()

                # print("-----------------------------")

                # 断开服务器链接
            robot.disconnect()

    except KeyboardInterrupt:
        robot.move_stop()

    except RobotError as e:
        logger.error("robot Event:{0}".format(e))

    finally:
        # 断开服务器链接
        if robot.connected:
            # 断开机械臂链接
            robot.disconnect()
        # 释放库资源
        Auboi5Robot.uninitialize()
        print("run end-------------------------")

# if __name__ == '__main__':
#     #test_process_demo()
#     test_rsm()
#     logger.info("test completed")



