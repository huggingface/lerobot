#!/usr/bin/env python3
"""
Simple Motor Control Interface with PID Visualization

This script provides a simple interface to control Feetech motors with real-time PID visualization.
Uses a dedicated motor communication thread for all motor I/O.
"""

import sys
import time
import select
import tty
import termios
import threading
import argparse
from typing import Dict
from collections import deque

import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QLineEdit
from PyQt6.QtCore import QTimer, pyqtSignal, QObject
import pyqtgraph as pg
from pglive.sources.data_connector import DataConnector
from pglive.sources.live_plot import LiveLinePlot
from pglive.sources.live_plot_widget import LivePlotWidget

from lerobot.common.motors.feetech import FeetechMotorsBus
from lerobot.common.motors.motors_bus import Motor, MotorNormMode


class MotorThread(threading.Thread):
    """Dedicated thread for motor communication at 250Hz."""
    
    def __init__(self, motor_controller):
        super().__init__(daemon=True)
        self.motor_controller = motor_controller
        self.running = False
        self.frequency = 250  # 250 Hz
        self.period = 1.0 / self.frequency
        
    def run(self):
        """Main motor communication loop."""
        self.running = True
        print(f"✓ Motor thread started at {self.frequency}Hz")
        
        while self.running:
            start_time = time.time()
            
            try:
                # Read current positions from motors
                current_positions = self.motor_controller.bus.sync_read("Present_Position", normalize=False)
                
                # Update cached current positions
                with self.motor_controller.data_lock:
                    self.motor_controller.cached_current_positions.update(current_positions)
                    self.motor_controller.cached_timestamp = time.time()
                
                # Write target positions to motors (from cache)
                with self.motor_controller.data_lock:
                    target_positions = self.motor_controller.cached_target_positions.copy()
                
                if target_positions:
                    self.motor_controller.bus.sync_write("Goal_Position", target_positions, normalize=False)
                    
            except Exception as e:
                print(f"Motor thread error: {e}")
                # Continue running even if communication fails
            
            # Maintain precise timing
            elapsed = time.time() - start_time
            sleep_time = max(0, self.period - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def stop(self):
        """Stop the motor thread."""
        self.running = False
        print("✓ Motor thread stopped")


class PIDVisualizer(QMainWindow):
    """Real-time PID visualization window."""
    
    def __init__(self, motor_controller, update_freq_hz=200):
        super().__init__()
        self.motor_controller = motor_controller
        self.update_freq_hz = update_freq_hz
        self.update_period_ms = int(1000 / update_freq_hz)
        self.dt = 1.0 / update_freq_hz
        
        self.setWindowTitle(f"Real-time PID Visualization ({update_freq_hz}Hz)")
        self.setGeometry(100, 100, 1200, 800)
        
        # Data storage
        self.max_points = 1000
        self.time_data = deque(maxlen=self.max_points)
        self.target_data = deque(maxlen=self.max_points)
        self.current_data = deque(maxlen=self.max_points)
        self.error_data = deque(maxlen=self.max_points)
        self.p_term_data = deque(maxlen=self.max_points)
        self.i_term_data = deque(maxlen=self.max_points)
        self.d_term_data = deque(maxlen=self.max_points)
        
        self.start_time = time.time()
        self.last_error = 0
        self.integral_sum = 0
        
        self.setup_ui()
        self.setup_timer()
    
    def setup_ui(self):
        """Set up the user interface with plots."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Status label
        self.status_label = QLabel("PID Visualization - Press 'a'/'d' to move motors")
        layout.addWidget(self.status_label)
        
        # Position input section
        input_layout = QHBoxLayout()
        input_layout.addWidget(QLabel("Go to position:"))
        self.position_input = QLineEdit()
        self.position_input.setPlaceholderText("Enter position (0-4095) and press Enter")
        self.position_input.returnPressed.connect(self.go_to_position)
        input_layout.addWidget(self.position_input)
        layout.addLayout(input_layout)
        
        # Create plots
        plots_layout = QVBoxLayout()
        
        # Position tracking plot
        self.position_widget = LivePlotWidget(title="Position Tracking")
        self.position_widget.x_range_controller.crop = True
        self.position_widget.x_range_controller.window_size = 30  # 30 second window
        
        self.target_plot = LiveLinePlot(pen="red", name="Target")
        self.current_plot = LiveLinePlot(pen="blue", name="Current")
        self.position_widget.addItem(self.target_plot)
        self.position_widget.addItem(self.current_plot)
        
        plots_layout.addWidget(self.position_widget)
        
        # Error plot
        self.error_widget = LivePlotWidget(title="Position Error")
        self.error_widget.x_range_controller.crop = True
        self.error_widget.x_range_controller.window_size = 30
        
        self.error_plot = LiveLinePlot(pen="orange", name="Error")
        self.error_widget.addItem(self.error_plot)
        
        plots_layout.addWidget(self.error_widget)
        
        # PID terms plot
        self.pid_widget = LivePlotWidget(title="PID Terms")
        self.pid_widget.x_range_controller.crop = True
        self.pid_widget.x_range_controller.window_size = 30
        
        self.p_plot = LiveLinePlot(pen="green", name="P term")
        self.i_plot = LiveLinePlot(pen="purple", name="I term")
        self.d_plot = LiveLinePlot(pen="cyan", name="D term")
        self.pid_widget.addItem(self.p_plot)
        self.pid_widget.addItem(self.i_plot)
        self.pid_widget.addItem(self.d_plot)
        
        plots_layout.addWidget(self.pid_widget)
        
        layout.addLayout(plots_layout)
    
    def setup_timer(self):
        """Set up timer for real-time updates."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(self.update_period_ms)  # Use configurable frequency
        
        # Add some initial data points to make plots visible
        initial_time = 0
        initial_pos = 2048  # Typical center position for STS motors
        self.time_data.append(initial_time)
        self.target_data.append(initial_pos)
        self.current_data.append(initial_pos)
        self.error_data.append(0)
        self.p_term_data.append(0)
        self.i_term_data.append(0)
        self.d_term_data.append(0)
        
        # Set initial data on plots
        self.target_plot.setData([initial_time], [initial_pos])
        self.current_plot.setData([initial_time], [initial_pos])
        self.error_plot.setData([initial_time], [0])
        self.p_plot.setData([initial_time], [0])
        self.i_plot.setData([initial_time], [0])
        self.d_plot.setData([initial_time], [0])
        
        print("✓ Timer started and plots initialized")
    
    def go_to_position(self):
        """Handle position input from the text field."""
        position_text = self.position_input.text().strip()
        
        if not position_text:
            return
        
        position = int(position_text)
        
        # Validate position range (typical for STS motors)
        if position < 0 or position > 4095:
            self.status_label.setText(f"❌ Invalid position {position}. Range: 0-4095")
            self.position_input.selectAll()  # Select text for easy correction
            return
        
        # Set target position in cache (motor thread will handle the actual write)
        motor_names = list(self.motor_controller.bus.motors.keys())
        with self.motor_controller.data_lock:
            for motor in motor_names:
                self.motor_controller.cached_target_positions[motor] = position
        
        # Update status and clear input
        self.status_label.setText(f"→ Moving to position {position}...")
        self.position_input.clear()
        print(f"✓ Set target position to {position}")
    
    def update_plots(self):
        """Update all plots with new data."""
        current_time = time.time() - self.start_time
        
        # Get motor data from thread-safe cache (no direct motor bus access)
        motor_name = list(self.motor_controller.bus.motors.keys())[0]  # Use first motor
        
        with self.motor_controller.data_lock:
            current_positions = self.motor_controller.cached_current_positions.copy()
            target_positions = self.motor_controller.cached_target_positions.copy()
        
        # Skip update if cache is empty (no data yet)
        if not current_positions or motor_name not in current_positions:
            return
            
        current_pos = current_positions[motor_name]
        target_pos = target_positions.get(motor_name, current_pos)
        
        # Calculate error and PID terms
        error = target_pos - current_pos
        
        # P term
        kp = self.motor_controller.cached_kp
        p_term = kp * error
        
        # I term (simple integration)
        self.integral_sum += error * self.dt
        ki = self.motor_controller.cached_ki
        i_term = ki * self.integral_sum
        
        # D term
        kd = self.motor_controller.cached_kd
        d_term = kd * (error - self.last_error) / self.dt
        self.last_error = error
        
        # Store data
        self.time_data.append(current_time)
        self.target_data.append(target_pos)
        self.current_data.append(current_pos)
        self.error_data.append(error)
        self.p_term_data.append(p_term)
        self.i_term_data.append(i_term)
        self.d_term_data.append(d_term)
        
        # Update plots with explicit data
        time_array = np.array(self.time_data)
        
        self.target_plot.setData(time_array, np.array(self.target_data))
        self.current_plot.setData(time_array, np.array(self.current_data))
        self.error_plot.setData(time_array, np.array(self.error_data))
        self.p_plot.setData(time_array, np.array(self.p_term_data))
        self.i_plot.setData(time_array, np.array(self.i_term_data))
        self.d_plot.setData(time_array, np.array(self.d_term_data))
        
        # Force plot updates
        self.position_widget.plotItem.getViewBox().autoRange()
        self.error_widget.plotItem.getViewBox().autoRange()
        self.pid_widget.plotItem.getViewBox().autoRange()
        
        # Update status
        self.status_label.setText(
            f"Position: {current_pos} | Target: {target_pos} | Error: {error:+.1f} | "
            f"P: {p_term:.1f} | I: {i_term:.1f} | D: {d_term:.1f} | "
            f"Points: {len(self.time_data)}"
        )


class SimpleMotorController:
    def __init__(self, port: str, motors: Dict[str, Motor], kp: float = 0.01, ki: float = 0.0001, kd: float = 0.001, 
                 enable_visualization: bool = True, print_pose: bool = True, plot_frequency: int = 200):
        """
        Initialize the motor controller.
        
        Args:
            port: Serial port for motor communication (e.g., "/dev/ttyUSB0")
            motors: Dictionary of motor name -> Motor object
            kp: Proportional gain coefficient
            ki: Integral gain coefficient  
            kd: Derivative gain coefficient
            enable_visualization: Whether to show PID visualization window
            print_pose: Whether to print real-time pose information
            plot_frequency: Plot update frequency in Hz
        """
        self.bus = FeetechMotorsBus(port=port, motors=motors)
        self.step_size = 50  # Movement step size in motor units
        self.connected = False
        self.enable_visualization = enable_visualization
        self.print_pose = print_pose
        self.plot_frequency = plot_frequency
        self.qt_app = None
        self.visualizer = None
        
        # Thread-safe motor data cache
        self.data_lock = threading.Lock()
        self.cached_current_positions = {}
        self.cached_target_positions = {}
        self.cached_timestamp = time.time()
        
        # Cached PID coefficients and other motor parameters
        self.cached_kp = kp
        self.cached_ki = ki  
        self.cached_kd = kd
        
        # Position input state
        self.input_state = "normal"  # "normal" or "position_input"
        self.position_buffer = ""
        
        # Motor communication thread
        self.motor_thread = None
        
    def connect(self):
        """Connect to the motors and configure them."""
        print("Connecting to motors...")
        self.bus.connect()
        self.bus.configure_motors()
        
        # Read and cache initial motor parameters
        self.update_cached_parameters()
        
        # Initialize target positions to current positions
        current_positions = self.bus.sync_read("Present_Position", normalize=False)
        with self.data_lock:
            self.cached_current_positions = current_positions.copy()
            self.cached_target_positions = current_positions.copy()
        
        self.connected = True
        print("✓ Motors connected and configured!")
        
        # Start motor communication thread
        self.motor_thread = MotorThread(self)
        self.motor_thread.start()
        
        # Show current positions
        self.show_current_positions()
        
        # Start visualization by default if enabled
        if self.enable_visualization:
            self.start_visualization()
    
    def update_cached_parameters(self):
        """Read current motor parameters and set user-specified PID values."""
        print("\nReading current motor parameters...")
        
        # First, read and display current values for information
        for motor_name in self.bus.motors.keys():
            try:
                p_coeff = self.bus.read("P_Coefficient", motor_name, normalize=False)
                i_coeff = self.bus.read("I_Coefficient", motor_name, normalize=False)  
                d_coeff = self.bus.read("D_Coefficient", motor_name, normalize=False)
                print(f"  {motor_name} current: P={p_coeff}, I={i_coeff}, D={d_coeff}")
                
            except Exception as e:
                print(f"  {motor_name}: Error reading current parameters - {e}")
        
        # Write the user-specified PID coefficients (from constructor args)
        self.set_pid_coefficients()
        print()
    
    def set_pid_coefficients(self):
        """Set PID coefficients to the user-specified values."""
        print(f"Setting PID coefficients to user values:")
        print(f"  Target values: P={self.cached_kp}, I={self.cached_ki}, D={self.cached_kd}")
        
        for motor_name in self.bus.motors.keys():
            try:
                # Convert to integers for motor (PID values must be integers 0-255)
                new_p = max(0, min(255, int(self.cached_kp)))
                new_i = max(0, min(255, int(self.cached_ki)))
                new_d = max(0, min(255, int(self.cached_kd)))
                
                # Write new PID values to motor
                self.bus.write("P_Coefficient", motor_name, new_p, normalize=False)
                self.bus.write("I_Coefficient", motor_name, new_i, normalize=False)
                self.bus.write("D_Coefficient", motor_name, new_d, normalize=False)
                print(f"  {motor_name}: Set to P={new_p}, I={new_i}, D={new_d}")
                
            except Exception as e:
                print(f"  {motor_name}: Error setting PID - {e}")
        
        # Read back and verify the values were set correctly
        print("Verifying written values:")
        for motor_name in self.bus.motors.keys():
            try:
                p_verify = self.bus.read("P_Coefficient", motor_name, normalize=False)
                i_verify = self.bus.read("I_Coefficient", motor_name, normalize=False)  
                d_verify = self.bus.read("D_Coefficient", motor_name, normalize=False)
                print(f"  {motor_name} verified: P={p_verify}, I={i_verify}, D={d_verify}")
                
            except Exception as e:
                print(f"  {motor_name}: Error verifying parameters - {e}")
    
    def disconnect(self):
        """Disconnect from motors safely."""
        if self.motor_thread:
            self.motor_thread.stop()
            self.motor_thread.join(timeout=1.0)
        
        self.bus.disable_torque()
        self.bus.disconnect()
        print("✓ Motors disconnected safely")
        self.connected = False
    
    def show_current_positions(self):
        """Display current motor positions from cache."""
        with self.data_lock:
            positions = self.cached_current_positions.copy()
        
        print("\nCurrent motor positions:")
        for motor_name, position in positions.items():
            print(f"  {motor_name}: {position}")
        print()
    
    def move_left(self):
        """Move all motors left (decrease target position in cache)."""
        with self.data_lock:
            current_targets = self.cached_target_positions.copy()
        
        # Update target positions in cache
        new_targets = {}
        for motor, target_pos in current_targets.items():
            new_targets[motor] = target_pos - self.step_size
        
        with self.data_lock:
            self.cached_target_positions.update(new_targets)
        
        print(f"← Moving left by {self.step_size} steps")
    
    def move_right(self):
        """Move all motors right (increase target position in cache)."""
        with self.data_lock:
            current_targets = self.cached_target_positions.copy()
        
        # Update target positions in cache  
        new_targets = {}
        for motor, target_pos in current_targets.items():
            new_targets[motor] = target_pos + self.step_size
        
        with self.data_lock:
            self.cached_target_positions.update(new_targets)
        
        print(f"→ Moving right by {self.step_size} steps")
    
    def handle_position_input_key(self, key):
        """Handle keyboard input during position input mode."""
        if key == '\r' or key == '\n':  # Enter key
            if self.position_buffer:
                position = int(self.position_buffer)
                if 0 <= position <= 4095:
                    self.go_to_position(position)
                    print(f"→ Moving to position {position}")
                else:
                    print(f"❌ Invalid position {position}. Range: 0-4095")
            self.input_state = "normal"
            self.position_buffer = ""
            print("\nBack to normal mode")
            
        elif key == '\x1b':  # Escape key
            self.input_state = "normal"
            self.position_buffer = ""
            print("\nPosition input cancelled")
            
        elif key.isdigit():
            self.position_buffer += key
            print(f"\rPosition: {self.position_buffer}", end="", flush=True)
            
        elif key == '\x7f' or key == '\b':  # Backspace
            if self.position_buffer:
                self.position_buffer = self.position_buffer[:-1]
                print(f"\rPosition: {self.position_buffer} ", end="", flush=True)
    
    def handle_normal_key(self, key):
        """Handle keyboard input during normal mode."""
        if key.lower() == 'q':
            return "quit"
        elif key.lower() == 'a':
            self.move_left()
        elif key.lower() == 'd':
            self.move_right()
        elif key.lower() == 'p':
            self.input_state = "position_input"
            self.position_buffer = ""
            print("\nEnter position (0-4095), press Enter to confirm, Esc to cancel:")
            print("Position: ", end="", flush=True)
        elif key.lower() == 'v' and not self.enable_visualization:
            self.start_visualization()
            self.enable_visualization = True
        elif key == '+' or key == '=':
            self.step_size = min(self.step_size + 10, 500)
            print(f"Step size: {self.step_size}")
        elif key == '-':
            self.step_size = max(self.step_size - 10, 10)
            print(f"Step size: {self.step_size}")
        return None
    
    def go_to_position(self, position):
        """Move all motors to specified position (update cache)."""
        motor_names = list(self.bus.motors.keys())
        
        with self.data_lock:
            for motor in motor_names:
                self.cached_target_positions[motor] = position
        
        print(f"→ Moving to position {position}")
    
    def start_visualization(self):
        """Start the Qt application and visualization window."""
        print("✓ Starting PID Visualization...")
        print("  - Position Tracking (red=target, blue=current)")
        print("  - Position Error (orange)")
        print("  - PID Terms (green=P, purple=I, cyan=D)")
        
        # Initialize Qt app in main thread if not already done
        if self.qt_app is None:
            self.qt_app = QApplication([])
            
        # Create and show the visualizer window  
        self.visualizer = PIDVisualizer(self, update_freq_hz=self.plot_frequency)
        self.visualizer.show()
        self.visualizer.raise_()  # Bring window to front
        self.visualizer.activateWindow()  # Make it active
        
        # Process Qt events to ensure window appears
        self.qt_app.processEvents()
        
        print("✓ PID Visualization window created!")
    
    def get_key(self):
        """Get a single keypress (non-blocking) - terminal already in raw mode."""
        import os
        
        # Check if input is available
        if select.select([sys.stdin], [], [], 0) == ([], [], []):
            return None
        
        # Input is available, read it directly (terminal already in raw mode)
        fd = sys.stdin.fileno()
        key = os.read(fd, 1).decode('utf-8', errors='ignore')
        
        return key
    
    def run_interface(self):
        """Run the main control interface with continuous real-time position display."""
        print("\n" + "="*80)
        print("SIMPLE MOTOR CONTROL INTERFACE - WITH PID VISUALIZATION")
        print("="*80)
        viz_status = "ON" if self.enable_visualization else "OFF"
        pose_status = "ON" if self.print_pose else "OFF"
        print(f"Visualization: {viz_status} | Pose Printing: {pose_status}")
        controls = "'a'=LEFT | 'd'=RIGHT | 'p'=Position | '+'=Inc Step | '-'=Dec Step"
        if not self.enable_visualization:
            controls += " | 'v'=Start Viz"
        controls += " | 'q'=QUIT"
        print(f"Controls: {controls}")
        print("="*80)
        
        # Set terminal to raw mode for the entire session
        fd = sys.stdin.fileno()
        self.old_terminal_settings = termios.tcgetattr(fd)
        tty.setraw(fd)
        
        # Initialize display state
        last_message = ""
        
        while True:
            # Handle pose display if enabled (using cached data)
            if self.print_pose and self.input_state == "normal":
                with self.data_lock:
                    current_positions = self.cached_current_positions.copy()
                    target_positions = self.cached_target_positions.copy()
                
                # Clear previous display
                if last_message:
                    lines_to_clear = last_message.count('\n') + 1
                    for _ in range(lines_to_clear):
                        print("\033[A\033[K", end="")  # Move up and clear line
                
                # Build display message
                message_lines = []
                message_lines.append(f"Step Size: {self.step_size:<3} | Motors: {len(self.bus.motors)} | PID: P={self.cached_kp} I={self.cached_ki} D={self.cached_kd}")
                message_lines.append("-" * 80)
                message_lines.append(f"{'MOTOR':<15} | {'TARGET':>6} | {'CURRENT':>6} | {'ERROR':>6} | {'STATUS':<10}")
                message_lines.append("-" * 80)
                
                for motor_name in self.bus.motors.keys():
                    target = target_positions.get(motor_name, 0)
                    current = current_positions.get(motor_name, 0)
                    error = target - current
                    
                    # Determine status
                    if abs(error) <= 2:
                        status = "ON TARGET"
                    elif abs(error) <= 10:
                        status = "CLOSE"
                    else:
                        status = "MOVING"
                    
                    message_lines.append(f"{motor_name:<15} | {target:>6} | {current:>6} | {error:>+6} | {status:<10}")
                
                message_lines.append("-" * 80)
                
                # Print the message
                last_message = '\n'.join(message_lines)
                print(last_message)
            
            # Check for keyboard input (non-blocking)
            key = self.get_key()
            
            # Process Qt events if visualization is running
            if self.qt_app is not None:
                self.qt_app.processEvents()
            
            # Handle keyboard input based on current state
            if key:
                if self.input_state == "normal":
                    result = self.handle_normal_key(key)
                    if result == "quit":
                        if self.print_pose:
                            # Clear the display before quitting
                            lines_to_clear = last_message.count('\n') + 1
                            for _ in range(lines_to_clear):
                                print("\033[A\033[K", end="")
                        print("\nQuitting...")
                        # Restore terminal settings before quitting
                        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self.old_terminal_settings)
                        break
                        
                elif self.input_state == "position_input":
                    self.handle_position_input_key(key)
            
            time.sleep(0.01)  # Update 100 times per second for responsive Qt events
        
        self.disconnect()
    
    def show_help(self):
        """Show help information."""
        print("\nAvailable commands:")
        print("  a - Move motors left (decrease position)")
        print("  d - Move motors right (increase position)")
        print("  p - Enter specific position (0-4095)")
        print("  + - Increase step size")
        print("  - - Decrease step size")
        print("  v - Start PID visualization")
        print("  q - Quit the program")
        print("Note: Real-time positions are always displayed!")
        print()


def main():
    """Main function - configure your motors here."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simple Motor Control Interface with PID Visualization")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial port for motor communication")
    parser.add_argument("--kp", type=float, default=1, help="Proportional gain coefficient")
    parser.add_argument("--ki", type=float, default=0, help="Integral gain coefficient")
    parser.add_argument("--kd", type=float, default=0, help="Derivative gain coefficient")
    parser.add_argument("--viz", default=True, help="Enable PID visualization window")
    parser.add_argument("--pose", default=False, help="Enable real-time pose printing")
    parser.add_argument("--freq", type=int, default=200, help="Plot update frequency in Hz (default: 200)")
    
    args = parser.parse_args()
    
    # Define your motors here - modify according to your setup
    # Format: {"motor_name": Motor(id=motor_id, model="motor_model", norm_mode=norm_mode)}
    motors = {
        "motor1": Motor(id=1, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
        # Add more motors as needed:
        # "motor2": Motor(id=2, model="sts3215", norm_mode=MotorNormMode.RANGE_0_100),
    }
    
    print("Simple Motor Control Interface with PID Visualization")
    print(f"Port: {args.port}")
    print(f"Motors: {list(motors.keys())}")
    print(f"PID coefficients: P={args.kp}, I={args.ki}, D={args.kd}")
    print(f"Visualization: {'Enabled' if args.viz else 'Disabled'}")
    print(f"Pose printing: {'Enabled' if args.pose else 'Disabled'}")
    
    # Create and run controller
    controller = SimpleMotorController(args.port, motors, kp=args.kp, ki=args.ki, kd=args.kd, 
                                     enable_visualization=args.viz, 
                                     print_pose=args.pose,
                                     plot_frequency=args.freq)
    controller.connect()
    controller.run_interface()


if __name__ == "__main__":
    main() 