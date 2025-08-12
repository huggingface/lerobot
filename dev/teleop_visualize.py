import signal
import sys
import time
from typing import Dict
from collections import deque

import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel
from PyQt6.QtCore import QTimer
import pyqtgraph as pg

from lerobot.common.teleoperators.so101_leader import SO101LeaderConfig, SO101Leader
from lerobot.common.robots.so101_follower import SO101FollowerConfig, SO101Follower

robot_config = SO101FollowerConfig(
    port="/dev/ttyACM0",
    id="follower10",
)

teleop_config = SO101LeaderConfig(
    port="/dev/ttyACM1",
    id="leader10",
)

robot = SO101Follower(robot_config)
teleop_device = SO101Leader(teleop_config)

# Global data sharing between main loop and visualization
class DataSharing:
    def __init__(self):
        self.motor_names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        self.leader_positions = {motor: 2048 for motor in self.motor_names}
        self.follower_positions = {motor: 2048 for motor in self.motor_names}
        self.timestamp = time.time()
        self.lock = False  # Simple lock to avoid reading while writing

shared_data = DataSharing()

class TeleopVisualizer(QMainWindow):
    """Real-time teleoperation visualization window."""
    
    def __init__(self, update_freq_hz=20):
        super().__init__()
        self.update_freq_hz = update_freq_hz
        self.update_period_ms = int(1000 / update_freq_hz)
        
        self.setWindowTitle(f"SO101 Teleoperation Visualization ({update_freq_hz}Hz)")
        self.setGeometry(100, 100, 1400, 900)
        
        # Data storage for plotting - 10 seconds of data
        self.recording_length_seconds = 10
        self.max_points = self.update_freq_hz * self.recording_length_seconds  # 20Hz * 10s = 200 points
        self.time_data = deque(maxlen=self.max_points)
        
        # Motor data storage
        self.motor_names = shared_data.motor_names
        self.leader_data = {motor: deque(maxlen=self.max_points) for motor in self.motor_names}
        self.follower_data = {motor: deque(maxlen=self.max_points) for motor in self.motor_names}
        self.error_data = {motor: deque(maxlen=self.max_points) for motor in self.motor_names}
        
        self.start_time = time.time()
        
        self.setup_ui()
        self.setup_timer()
    
    def setup_ui(self):
        """Set up the user interface with plots."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Status label
        self.status_label = QLabel("SO101 Teleoperation - Leader (red) vs Follower (blue)")
        layout.addWidget(self.status_label)
        
        # Create plots layout
        plots_layout = QVBoxLayout()
        
        # Individual motor plots (2x3 grid)
        motor_plots_layout = QVBoxLayout()
        
        # Top row: shoulder_pan, shoulder_lift, elbow_flex
        top_row = QHBoxLayout()
        self.motor_widgets = {}
        self.leader_plots = {}
        self.follower_plots = {}
        self.error_plots = {}
        
        for i, motor in enumerate(self.motor_names[:3]):
            widget = pg.PlotWidget(title=f"{motor.replace('_', ' ').title()}")
            widget.setLabel('left', 'Position')
            widget.setLabel('bottom', 'Time (s)')
            widget.addLegend()
            
            # Create plot data items
            leader_plot = widget.plot([], [], pen='r', name="Leader")
            follower_plot = widget.plot([], [], pen='b', name="Follower")
            
            self.motor_widgets[motor] = widget
            self.leader_plots[motor] = leader_plot
            self.follower_plots[motor] = follower_plot
            
            top_row.addWidget(widget)
        
        motor_plots_layout.addLayout(top_row)
        
        # Bottom row: wrist_flex, wrist_roll, gripper
        bottom_row = QHBoxLayout()
        for motor in self.motor_names[3:]:
            widget = pg.PlotWidget(title=f"{motor.replace('_', ' ').title()}")
            widget.setLabel('left', 'Position')
            widget.setLabel('bottom', 'Time (s)')
            widget.addLegend()
            
            # Create plot data items
            leader_plot = widget.plot([], [], pen='r', name="Leader")
            follower_plot = widget.plot([], [], pen='b', name="Follower")
            
            self.motor_widgets[motor] = widget
            self.leader_plots[motor] = leader_plot
            self.follower_plots[motor] = follower_plot
            
            bottom_row.addWidget(widget)
        
        motor_plots_layout.addLayout(bottom_row)
        plots_layout.addLayout(motor_plots_layout)
        
        # Combined tracking error plot
        self.error_widget = pg.PlotWidget(title="Tracking Errors (Leader - Follower)")
        self.error_widget.setLabel('left', 'Error')
        self.error_widget.setLabel('bottom', 'Time (s)')
        self.error_widget.addLegend()
        
        # Different colors for each motor error
        colors = ['r', 'g', 'b', 'orange', 'm', 'c']
        for i, motor in enumerate(self.motor_names):
            error_plot = self.error_widget.plot([], [], pen=colors[i], name=motor.replace('_', ' '))
            self.error_plots[motor] = error_plot
        
        plots_layout.addWidget(self.error_widget)
        
        layout.addLayout(plots_layout)

    def setup_timer(self):
        """Set up timer for real-time updates."""
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_plots)
        self.timer.start(self.update_period_ms)
        
        print("✓ Visualization timer started")
    
    def update_plots(self):
        """Update all plots with new data from shared_data."""
        if shared_data.lock:
            return  # Skip if data is being updated
            
        current_time = time.time() - self.start_time
        
        # Store time data
        self.time_data.append(current_time)
        
        # Use shared data from main loop (no motor reading here!)
        errors = {}
        for motor in self.motor_names:
            leader_pos = shared_data.leader_positions.get(motor, 2048)
            follower_pos = shared_data.follower_positions.get(motor, 2048)
            error = leader_pos - follower_pos
            
            self.leader_data[motor].append(leader_pos)
            self.follower_data[motor].append(follower_pos)
            self.error_data[motor].append(error)
            
            errors[motor] = error
        
        # Update plots using PyQtGraph's setData method
        time_array = list(self.time_data)
        
        for motor in self.motor_names:
            # Update individual motor plots
            leader_array = list(self.leader_data[motor])
            follower_array = list(self.follower_data[motor])
            error_array = list(self.error_data[motor])
            
            self.leader_plots[motor].setData(time_array, leader_array)
            self.follower_plots[motor].setData(time_array, follower_array)
            self.error_plots[motor].setData(time_array, error_array)
        
        # Update status with current errors
        max_error = max(abs(e) for e in errors.values())
        avg_error = sum(abs(e) for e in errors.values()) / len(errors)
        
        self.status_label.setText(
            f"Max Error: {max_error:.1f} | Avg Error: {avg_error:.1f} | "
            f"Errors: SP:{errors['shoulder_pan']:+.0f} SL:{errors['shoulder_lift']:+.0f} "
            f"EF:{errors['elbow_flex']:+.0f} WF:{errors['wrist_flex']:+.0f} "
            f"WR:{errors['wrist_roll']:+.0f} GR:{errors['gripper']:+.0f}"
        )


def cleanup_and_exit(signum=None, frame=None):
    """Disable torque and disconnect motors on exit."""
    print("\nDisabling torque on both devices")
    robot.bus.disable_torque()
    teleop_device.bus.disable_torque()
    robot.disconnect()
    teleop_device.disconnect()
    print("✓ Motors disabled")
    sys.exit(0)

def main():
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, cleanup_and_exit)

    print("=== SO101 TELEOPERATION WITH VISUALIZATION ===")
    print("Connecting devices...")
    
    robot.connect(calibrate=True)
    teleop_device.connect(calibrate=True)
    
    print("✓ Both devices connected")
    
    # Start Qt application for visualization
    print("Starting visualization...")
    qt_app = QApplication([])
    
    # Create and show visualizer
    visualizer = TeleopVisualizer(update_freq_hz=20)
    visualizer.show()
    visualizer.raise_()
    visualizer.activateWindow()
    
    print("✓ Visualization started")
    print("Teleoperation active. Move the leader arm to see real-time tracking.")
    print("Press Ctrl+C to exit.")
    
    # Main teleoperation loop (like teleop.py)
    try:
        while True:
            # Read raw positions from leader without normalization  
            shared_data.lock = True
            try:
                raw_positions = teleop_device.bus.sync_read("Present_Position", normalize=False)
                # Send raw positions to follower
                robot.bus.sync_write("Goal_Position", raw_positions, normalize=False)
                
                # Also read follower positions for visualization
                follower_positions = robot.bus.sync_read("Present_Position", normalize=False)
                
                # Update shared data
                shared_data.leader_positions = raw_positions
                shared_data.follower_positions = follower_positions
                shared_data.timestamp = time.time()
            finally:
                shared_data.lock = False
                
            # Process Qt events to keep GUI responsive
            qt_app.processEvents()
            
    except KeyboardInterrupt:
        cleanup_and_exit()

if __name__ == "__main__":
    main()