
    #robot.arm_bus.write("Acceleration", [20], ["shoulder_pitch"])    

    ####DEBUGGER####################
    # joint = input("Enter joint name: ")
    # encoder = EncoderReader("/dev/ttyUSB1", 115200)
    # start_angle = arm_calibration['start_pos'][arm_calibration['motor_names'].index(joint)]
    # end_angle = arm_calibration['end_pos'][arm_calibration['motor_names'].index(joint)]
    # # start_angle = shoulder_calibration['start_pos'][shoulder_calibration['motor_names'].index(joint)]
    # # end_angle = shoulder_calibration['end_pos'][shoulder_calibration['motor_names'].index(joint)]
    # while True:
    #     angle = int(start_angle+(end_angle-start_angle)*encoder.read()/1000)
    #     # robot.shoulder_bus.set_bus_baudrate(115200)
    #     # robot.shoulder_bus.write("Goal_Position",angle, [joint])
    #     robot.shoulder_bus.set_bus_baudrate(1000000)
    #     robot.arm_bus.write("Goal_Position",angle, [joint])
    #     print(angle)
    #     time.sleep(0.1)

    

    #####SAFETY CHECKS EXPLAINED#####
    #There are two safety checks built-in: one is based on load and the other is based on current. 
    #Current: if Protection_Current > Present_Current we wait Over_Current_Protection_Time (expressed in ms) and set Torque_Enable to 0
    #Load: if Max_Torque_Limit*Overload_Torque (expressed as a percentage) > Present_Load, we wait Protection_Time (expressed in ms 
    #and set Max_Torque_Limit to Protective_Torque)
    #Though we can specify Min-Max_Angle_Limit, Max_Temperature_Limit, Min-Max_Voltage_Limit, no safety checks are implemented for these values

    #robot.arm_bus.set_calibration(arm_calibration)




    #method 1
    # robot.arm_bus.write("Overload_Torque", 80)
    # robot.arm_bus.write("Protection_Time", 10)
    # robot.arm_bus.write("Protective_Torque", 1)
    # robot.arm_bus.write("Protection_Current", 200,["shoulder_pitch"])
    # robot.arm_bus.write("Over_Current_Protection_Time", 10)
        
    #method 2
    # robot.arm_bus.write("Protection_Current", 500,["shoulder_pitch"])
    # robot.arm_bus.write("Over_Current_Protection_Time", 10)
    # robot.arm_bus.write("Max_Torque_Limit", 1000)
    # robot.arm_bus.write("Overload_Torque", 40)
    # robot.arm_bus.write("Protection_Time", 10)
    # robot.arm_bus.write("Protective_Torque", 1)

    # robot.shoulder_bus.set_bus_baudrate(115200)
    # robot.shoulder_bus.write("Goal_Position",2500)
    # exit()

    ######LOGGER####################
    # from test_torque.log_and_plot_feetech import log_and_plot_params 

    # params_to_log = [
    #     "Protection_Current",
    #     "Present_Current",
    #     "Max_Torque_Limit",
    #     "Protection_Time",
    #     "Overload_Torque",
    #     "Present_Load",
    #     "Present_Position",
    # ]

    # servo_names = ["shoulder_pitch"]
    
    
    # servo_data, timestamps = log_and_plot_params(robot.shoulder_bus, params_to_log, servo_names, test_id="shoulder_pitch")
    # exit()

    
    #robot.arm_bus.write("Goal_Position",2300, ["shoulder_pitch"])
    # dt = 2
    # steps = 4
    # max_pos = 1500
    # min_pos = 2300
    # increment = (max_pos - min_pos) / steps
    # # Move from min_pos to max_pos in steps
    # for i in range(steps + 1):  # Include the last step
    #     current_pos = min_pos + int(i * increment)
    #     robot.arm_bus.write("Goal_Position", [current_pos], ["shoulder_pitch"])
    #     time.sleep(dt)

    # # Move back from max_pos to min_pos in steps
    # for i in range(steps + 1):  # Include the last step
    #     current_pos = max_pos - int(i * increment)
    #     robot.arm_bus.write("Goal_Position", [current_pos], ["shoulder_pitch"])
    #     time.sleep(dt)shoulder_pitch
    #demo to show how sending a lot of values makes the robt shake 



    # # Step increment
    #

    # # Move from min_pos to max_pos in steps
    # for i in range(steps + 1):  # Include the last step
    #     current_pos = min_pos + int(i * increment)
    #     robot.arm_bus.write("Goal_Position", [current_pos], ["elbow_flex"])
    #     time.sleep(dt)

    # # Move back from max_pos to min_pos in steps
    # for i in range(steps + 1):  # Include the last step
    #     current_pos = max_pos - int(i * increment)
    #     robot.arm_bus.write("Goal_Position", [current_pos], ["elbow_flex"])
    #     time.sleep(dt)
    # exit()

    #robot.arm_bus.write("Goal_Position", a    # shoulder_calibration = robot.get_shoulder_calibration()
    # print(shoulder_calibration)m_calibration["start_pos"])
    # robot.arm_bus.write("Over_Current_Protection_Time", 50)
    # robot.arm_bus.write("Protection_Current", 310, ["shoulder_pitch"])
    # robot.arm_bus.write("Overload_Torque", 80, ["shoulder_pitch"])
    # robot.arm_bus.write("Protection_Time", 100, ["shoulder_pitch"])
    # robot.arm_bus.write("Over_Current_Protection_Time", 50, ["shoulder_pitch"])
    
    # robot.arm_bus.write("Protective_Torque", 20, ["shoulder_pitch"])
        
        
    # robot.arm_bus.write("Goal_Position", [600],["shoulder_pitch"])
    
    # from test_torque.log_and_plot_feetech import log_and_plot_params 

    # params_to_log = [
    #     "Present_Current",
    #     "Protection_Current",
    #     "Overload_Torque",
    #     "Protection_Time",
    #     "Protective_Torque",
    #     "Present_Load",
    #     "Present_Position",
    # ]

    # servo_names = ["shoulder_pitch"]
    
    # 

    #robot.arm_bus.write("Goal_Position", arm_calibration["start_pos"])

    #robot.hand_bus.set_calibration(hand_calibration)

    #interp = 0.3

    #robot.arm_bus.write("Goal_Position", [int((i*interp+j*(1-interp))) for i, j in zip(arm_calibration["start_pos"], arm_calibration["end_pos"])])
    #exit()

    # glove = HomonculusGlove()
    # glove.run_calibration()



####GOOD FOR GRASPING
        # start_pos = [
        #     500,
        #     900,
        #     500,
        #     1000,
        #     100,
        #     450,#250
        #     950,#750
        #     100,
        #     300,#400
        #     50,#150
        #     100,
        #     120,
        #     980,
        #     100,
        #     950,
        #     750,
        # ]
        # end_pos = [
        #     start_pos[0] - 400,
        #     start_pos[1] - 300,
        #     start_pos[2] + 500,
        #     start_pos[3] - 50,
        #     start_pos[4] + 900,
        #     start_pos[5] + 500,
        #     start_pos[6] - 500,
        #     start_pos[7] + 900,
        #     start_pos[8] + 700,
        #     start_pos[9] + 700,
        #     start_pos[10] + 900,
        #     start_pos[11] + 700,
        #     start_pos[12] - 700,
        #     start_pos[13] + 900,
        #     start_pos[14] - 700,
        #     start_pos[15] - 700,
        # ]






SCS_SERIES_CONTROL_TABLE = {

    # "Max_Torque_Limit": (16, 2),
    # "Phase": (18, 1),
    # "Unloading_Condition": (19, 1),

    "Protective_Torque": (37, 1),
    "Protection_Time": (38, 1),
    #Baud_Rate": (48, 1),

}

def read_and_print_scs_values(robot):
    for param_name in SCS_SERIES_CONTROL_TABLE:
        value = robot.hand_bus.read(param_name)
        print(f"{param_name}: {value}")

motor_1_values = {
    "Lock" : 255,
    #"Protection_Time": 20#so if you write to these they turn to 0 for some fucking reason. protection time was 100, procetive to
}

# motor_1_values = {
#     "Lock": 1,
#     "Protection_Time": 100,
#     "Protective_Torque": 20,
#     "Phase": 1,#thisu is bullshit
#     "Unloading_Condition": 32,

# }
#bug in writing to specific values of the scs0009

# Write values to motor 2, there is overload torque there 
#ok so i can write, the jittering is because of the overload torque which is still being triggered

#TODO: i have to write a functioining version for the sc009 (or i dont who cares)
