import math
import logging
import pygame
import time

class PS4JoystickController:
    def __init__(
        self,
        motor_names,
        initial_position=[90, 180, 180, 0, 0, 10],
        l1=117.0,  # Length of first lever in mm
        l2=136.0,  # Length of second lever in mm
        *args,
        **kwargs
    ):
        self.motor_names = motor_names
        self.initial_position = initial_position
        self.current_positions = {name: pos for name, pos in zip(self.motor_names, self.initial_position)}
        self.new_positions = self.current_positions.copy()

        # Inverse Kinematics parameters are used to compute x and y positions
        self.l1 = l1
        self.l2 = l2

        # x and y are coordinates of the axis of wrist_flex motor relative to the axis of the shoulder_pan motor in mm
        self.x, self.y = self._compute_position(self.current_positions['shoulder_lift'], self.current_positions['elbow_flex'])

        # Gamepad states
        self.axes = {
            'RX': 0.0,
            'RY': 0.0,
            'LX': 0.0,
            'LY': 0.0,
            'L2': 0.0,
            'R2': 0.0,
        }
        self.buttons = {
            'L2': 0,
            'R2': 0,
            'DPAD_LEFT': 0,
            'DPAD_RIGHT': 0,
            'DPAD_UP': 0,
            'DPAD_DOWN': 0,
            'X': 0,
            'O': 0,
            'T': 0,
            'S': 0,
            'L1': 0,
            'R1': 0,
            'SHARE': 0,
            'OPTIONS': 0,
            'PS': 0,
            'L3': 0,
            'R3': 0,
        }

        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            raise IOError("No joystick/gamepad found.")
        
        self.last_event = time.time()
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()

    def _deadzone(self, value, threshold=0.1):
        """
        Apply a deadzone to the joystick input to avoid drift.
        """
        if abs(value) < threshold:
            return 0.0
        return value

    def _reinitialize_joystick(self):
        """
        Reinitialize the joystick connection
        """
        # I often lose connection when use bluetooth joystick, TODO: fix this
        # Not a problem when connected via USB

        try:
            logging.warning("No events received, reinitializing joystick...")
            self.joystick.quit()
            pygame.joystick.quit()
            pygame.event.clear()

            pygame.joystick.init()
            self.joystick = pygame.joystick.Joystick(0)
            self.joystick.init()
            logging.info("Joystick reinitialized successfully")
        except Exception as e:
            logging.error(f"Failed to reinitialize joystick: {e}")

    def _read_inputs(self):
        """
        Read inputs from the PS4 joystick using direct state reading.
        """
        # pygame doesn't allow using threading on Mac for joystick events
        # Not really a problem as I can see (it is still fast) but there can be a better solution

        try:
            pygame.event.pump()

            # Read axes states
            self.axes['LX'] = self._deadzone(self.joystick.get_axis(0))
            self.axes['LY'] = self._deadzone(-self.joystick.get_axis(1))
            self.axes['RX'] = self._deadzone(self.joystick.get_axis(2))
            self.axes['RY'] = self._deadzone(-self.joystick.get_axis(3))
            self.axes['L2'] = self._deadzone((self.joystick.get_axis(4) + 1) / 2)
            self.axes['R2'] = self._deadzone((self.joystick.get_axis(5) + 1) / 2)

            # Read button states
            button_mappings = {
                0: 'X', 1: 'O', 2: 'S', 3: 'T',
                4: 'SHARE', 6: 'OPTIONS', 7: 'L3', 8: 'R3',
                9: 'L1', 10: 'R1', 11: 'DPAD_UP', 12: 'DPAD_DOWN',
                14: 'DPAD_LEFT', 13: 'DPAD_RIGHT'
            }
            
            for button_id, button_name in button_mappings.items():
                self.buttons[button_name] = self.joystick.get_button(button_id)

            # Process the current state
            self._process_inputs()

        except Exception as e:
            logging.error(f"Error reading gamepad inputs: {e}")
            self._reinitialize_joystick()

    def _process_inputs(self):
        # Compute new positions based on inputs
        SPEED = 0.3
        # TODO: speed can be different for different directions

        temp_positions = self.current_positions.copy()

        # Handle macro buttons
        # Buttons have assigned states where robot can move directly
        used_macros = False
        macro_buttons = ['X', 'O', 'T', 'S'] # can use more if needed
        for button in macro_buttons:
            if self.buttons.get(button):
                temp_positions = self._execute_macro(button, temp_positions)
                temp_x, temp_y = self._compute_position(temp_positions['shoulder_lift'], temp_positions['elbow_flex'])
                correct_inverse_kinematics = True
                used_macros = True

        if not used_macros:
            # Map joystick inputs to motor positions
            # Right joystick controls "wrist_roll" (left/right) and "wrist_flex" (up/down)
            temp_positions['wrist_roll'] += self.axes['RX'] * SPEED  # degrees per update
            temp_positions['wrist_flex'] -= self.axes['RY'] * SPEED  # degrees per update

            # L2 and R2 control gripper
            temp_positions['gripper'] -= SPEED * self.axes['R2']  # Close gripper
            temp_positions['gripper'] += SPEED * self.axes['L2'] # Open gripper

            # Left joystick and dpad left and right control shoulder_pan
            temp_positions['shoulder_pan'] += (self.axes['LX'] + self.buttons['DPAD_LEFT'] - self.buttons['DPAD_RIGHT']) * SPEED  # degrees per update

            # Handle the linear movement of the arm
            # Left joystick up/down changes x
            temp_x = self.x + self.axes['LY'] * SPEED  # mm per update

            # D-pad up/down change y
            temp_y = self.y + (self.buttons['DPAD_UP'] - self.buttons['DPAD_DOWN']) * SPEED

            correct_inverse_kinematics = False

            # Compute shoulder_lift and elbow_flex angles based on x and y
            try:
                temp_positions['shoulder_lift'], temp_positions['elbow_flex'] = self._compute_inverse_kinematics(temp_x, temp_y)
                shoulder_lift_change = temp_positions['shoulder_lift'] - self.current_positions['shoulder_lift']
                elbow_flex_change = temp_positions['elbow_flex'] - self.current_positions['elbow_flex']
                temp_positions['wrist_flex'] +=  shoulder_lift_change - elbow_flex_change
                correct_inverse_kinematics = True
            except ValueError as e:
                logging.error(f"Error computing inverse kinematics: {e}")

        # Perform eligibility check
        if self._is_position_valid(temp_positions, temp_x, temp_y) and correct_inverse_kinematics:
            # Atomic update: all positions are valid, apply the changes
            self.current_positions = temp_positions
            self.x = temp_x
            self.y = temp_y
        else:
            # Invalid positions detected, do not update
            logging.warning("Invalid motor positions detected. Changes have been discarded.")
            self.joystick.rumble(0.5, 0.5, 200)

    def _is_position_valid(self, positions, x, y):
        """
        Check if all positions are within their allowed ranges.
        Define the allowed ranges for each motor.
        """
        allowed_ranges = {
            'shoulder_pan': (-10, 190),
            'shoulder_lift': (-5, 185),
            'elbow_flex': (-5, 185),
            'wrist_flex': (-110, 110),
            'wrist_roll': (-110, 110),
            'gripper': (0, 100),
            'x': (15, 250),
            'y': (-110, 250),
        }

        for motor, (min_val, max_val) in allowed_ranges.items():
            if motor in positions:
                if not (min_val <= positions[motor] <= max_val):
                    logging.error(f"Motor '{motor}' position {positions[motor]} out of range [{min_val}, {max_val}].")
                    return False
                
        # Check if x and y positions are within the allowed ranges
        if x < allowed_ranges['x'][0] or x > allowed_ranges['x'][1]:
            logging.error(f"X position {x} out of range {allowed_ranges['x']}.")
            return False
    
        if y < allowed_ranges['y'][0] or y > allowed_ranges['y'][1]:
            logging.error(f"Y position {y} out of range {allowed_ranges['y']}.")
            return False

        return True

    def _execute_macro(self, button, positions):
        """
        Define macros for specific buttons. When a macro button is pressed,
        set the motors to predefined positions.
        """
        macros = {
            'O': [90, 180, 180, 0, 0, 10], # initial position
            'X': [90, 45, 135, -90, 90, 80], # low horizontal gripper
            'T': [90, 130, 150, 70, 90, 80], # top down gripper
            'S': [90, 160, 140, 20, 0, 0], # looking forward
            # can add more macros for all other buttons
        }

        if button in macros:
            motor_postitions = macros[button][:6]
            for name, pos in zip(self.motor_names, motor_postitions):
                positions[name] = pos
            logging.info(f"Macro '{button}' executed. Motors set to {motor_postitions}.")
        return positions

    def _compute_inverse_kinematics(self, x, y):
        """
        Compute motor 2 and motor 3 angles based on the desired x and y positions.
        """
        #TODO: add explanation of the math behind this
        #TODO: maybe the math can be optimized, check it

        l1 = self.l1
        l2 = self.l2

        # Compute the distance from motor 2 to the desired point
        distance = math.hypot(x, y)

        # Check if the point is reachable
        if distance > (l1 + l2) or distance < abs(l1 - l2):
            raise ValueError(f"Point ({x}, {y}) is out of reach.")

        # Compute angle for motor3 (theta2)
        cos_theta2 = (l1**2 + l2**2 - distance**2) / (2 * l1 * l2)
        # Clamp the value to avoid numerical issues
        cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
        theta2_rad = math.acos(cos_theta2)
        theta2_deg = math.degrees(theta2_rad)
        # Adjust motor3 angle

        offset =math.degrees(math.asin(32/l1))

        motor3_angle = 180 - (theta2_deg - offset)

        # Compute angle for motor2 (theta1)
        cos_theta1 = (l1**2 + distance**2 - l2**2) / (2 * l1 * distance)
        # Clamp the value to avoid numerical issues
        cos_theta1 = max(min(cos_theta1, 1.0), -1.0)
        theta1_rad = math.acos(cos_theta1)
        theta1_deg = math.degrees(theta1_rad)
        alpha_rad = math.atan2(y, x)
        alpha_deg = math.degrees(alpha_rad)

        beta_deg = 180 - alpha_deg - theta1_deg
        motor2_angle = 180 - beta_deg + offset

        return motor2_angle, motor3_angle
    
    def _compute_position(self, motor2_angle, motor3_angle):
        """
        Compute the x and y positions based on the motor 2 and motor 3 angles.
        """
        l1 = self.l1
        l2 = self.l2
        offset = math.degrees(math.asin(32/l1))

        beta_deg = 180 - motor2_angle + offset
        beta_rad = math.radians(beta_deg)

        theta2_deg = 180 - motor3_angle + offset
        theta2_rad = math.radians(theta2_deg)

        y = l1 * math.sin(beta_rad) - l2 * math.sin(beta_rad - theta2_rad)
        x = - l1 * math.cos(beta_rad) + l2 * math.cos(beta_rad - theta2_rad)
        return x, y

    def get_command(self):
        """
        Return the current motor positions after reading and processing inputs.
        """
        self._read_inputs()
        return self.current_positions.copy()

    def stop(self):
        """
        Clean up resources.
        """
        try:
            self.joystick.quit()
        except:
            pass
        pygame.quit()
