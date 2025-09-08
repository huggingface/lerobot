#!/usr/bin/env python3
"""
Servo Motor Debug UI for LeRobot Motors Bus
A tkinter-based debugging interface for servo motors using the lerobot motors_bus API
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import numpy as np
from collections import deque
import glob
import platform

from lerobot.motors.dynamixel import DynamixelMotorsBus
from lerobot.motors.feetech import FeetechMotorsBus


class ServoDebugUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Servo Motor Debug Tool")
        self.root.geometry("1000x700")
        
        # Motor bus instance (will be initialized when connected)
        self.motor_bus = None
        self.connected = False
        self.selected_motor_id = None
        self.monitoring = False
        self.monitor_thread = None
        
        # Data for plotting
        self.plot_data = {
            'position': deque(maxlen=100),
            'torque': deque(maxlen=100),
            'speed': deque(maxlen=100),
            'current': deque(maxlen=100),
            'temperature': deque(maxlen=100),
            'voltage': deque(maxlen=100)
        }
        
        self.setup_ui()
        
    def get_usb_serial_ports(self) -> List[str]:
        """
        Detect available USB serial ports on the system.
        Returns a list of port paths.
        """
        ports = []
        
        # For macOS and Linux
        if platform.system() in ['Darwin', 'Linux']:
            # Look for USB serial devices
            patterns = [
                '/dev/ttyUSB*',      # Linux USB-to-serial
                '/dev/ttyACM*',      # Linux CDC ACM devices
                '/dev/tty.usbserial*',  # macOS USB-to-serial
                '/dev/tty.usbmodem*',   # macOS USB modem devices
                '/dev/cu.usbserial*',   # macOS USB-to-serial (callout)
                '/dev/cu.usbmodem*'     # macOS USB modem (callout)
            ]
            
            for pattern in patterns:
                ports.extend(glob.glob(pattern))
        
        # For Windows (if needed in future)
        elif platform.system() == 'Windows':
            # Windows COM ports would be handled differently
            # For now, just return common Windows ports
            ports = [f'COM{i}' for i in range(1, 10)]
        
        # Remove duplicates and sort
        ports = sorted(list(set(ports)))
        
        # If no ports found, provide some defaults
        if not ports:
            if platform.system() == 'Darwin':
                ports = ['/dev/tty.usbmodem1', '/dev/tty.usbserial1']
            elif platform.system() == 'Linux':
                ports = ['/dev/ttyUSB0', '/dev/ttyUSB1']
            else:
                ports = ['COM1', 'COM2']
        
        return ports
        
    def setup_ui(self):
        # Main container
        main_frame = ttk.Frame(self.root, padding="5")
        main_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Left panel - Connection and Motor List
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        
        # Connection Settings
        conn_frame = ttk.LabelFrame(left_panel, text="Com Settings", padding="10")
        conn_frame.pack(fill="x", pady=(0, 10))
        
        ttk.Label(conn_frame, text="Com:").grid(row=0, column=0, sticky="w", padx=(0, 5))
        self.com_port = ttk.Combobox(conn_frame, width=20)
        self.com_port.grid(row=0, column=1, columnspan=2, padx=(0, 5))
        
        # Get available USB ports and set them
        self.refresh_com_ports()
        
        # Add refresh button for COM ports
        ttk.Button(conn_frame, text="↻", width=3, command=self.refresh_com_ports).grid(row=0, column=3, padx=(0, 5))
        
        ttk.Label(conn_frame, text="BaudR:").grid(row=1, column=0, sticky="w", padx=(0, 5))
        self.baudrate = ttk.Combobox(conn_frame, width=20)
        self.baudrate.grid(row=1, column=1, columnspan=2, padx=(0, 5))
        self.baudrate.set("1000000")
        self.baudrate['values'] = ['9600', '57600', '115200', '1000000', '2000000', '3000000', '4000000']
        
        ttk.Label(conn_frame, text="Timeout:").grid(row=2, column=0, sticky="w", padx=(0, 5))
        self.timeout = ttk.Entry(conn_frame, width=22)
        self.timeout.grid(row=2, column=1, columnspan=2, padx=(0, 5))
        self.timeout.insert(0, "50")
        
        self.connect_btn = ttk.Button(conn_frame, text="Open", command=self.toggle_connection)
        self.connect_btn.grid(row=3, column=0, columnspan=4, pady=(10, 0))
        
        # Motor List
        motor_frame = ttk.LabelFrame(left_panel, text="Servo List", padding="10")
        motor_frame.pack(fill="both", expand=True)
        
        self.search_btn = ttk.Button(motor_frame, text="Search", command=self.search_motors)
        self.search_btn.pack(fill="x", pady=(0, 5))
        
        # Motor list with columns
        columns = ('ID', 'Module')
        self.motor_tree = ttk.Treeview(motor_frame, columns=columns, height=10, show='headings')
        self.motor_tree.heading('ID', text='ID')
        self.motor_tree.heading('Module', text='Module')
        self.motor_tree.column('ID', width=50)
        self.motor_tree.column('Module', width=100)
        self.motor_tree.pack(fill="both", expand=True)
        self.motor_tree.bind('<<TreeviewSelect>>', self.on_motor_select)
        
        # Right panel - Tabbed interface
        right_panel = ttk.Notebook(main_frame)
        right_panel.grid(row=0, column=1, sticky="nsew")
        main_frame.columnconfigure(1, weight=1)
        
        # Debug Tab
        self.debug_tab = ttk.Frame(right_panel)
        right_panel.add(self.debug_tab, text="Debug")
        self.setup_debug_tab()
        
        # Programming Tab
        self.prog_tab = ttk.Frame(right_panel)
        right_panel.add(self.prog_tab, text="Programming")
        self.setup_programming_tab()
        
    def setup_debug_tab(self):
        # Control table frame
        table_frame = ttk.LabelFrame(self.debug_tab, text="Control Table", padding="5")
        table_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(table_frame)
        btn_frame.pack(fill="x", pady=(0, 5))
        
        ttk.Button(btn_frame, text="Save", command=self.save_config).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Load", command=self.load_config).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Online", command=self.go_online).pack(side="left", padx=2)
        ttk.Button(btn_frame, text="Recovery", command=self.recovery).pack(side="left", padx=2)
        
        # Control table
        columns = ('Address', 'Memory', 'Value', 'Area', 'R/W')
        self.control_table = ttk.Treeview(table_frame, columns=columns, height=20, show='headings')
        
        for col in columns:
            self.control_table.heading(col, text=col)
            self.control_table.column(col, width=100)
        
        scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.control_table.yview)
        self.control_table.configure(yscrollcommand=scroll.set)
        
        self.control_table.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")
        
        # Value edit frame
        edit_frame = ttk.Frame(table_frame)
        edit_frame.pack(fill="x", pady=5)
        
        ttk.Label(edit_frame, text="ID:").pack(side="left", padx=5)
        self.edit_id = ttk.Entry(edit_frame, width=10)
        self.edit_id.pack(side="left", padx=5)
        
        ttk.Label(edit_frame, text="Value:").pack(side="left", padx=5)
        self.edit_value = ttk.Entry(edit_frame, width=10)
        self.edit_value.pack(side="left", padx=5)
        
        ttk.Button(edit_frame, text="Set", command=self.set_value).pack(side="left", padx=5)
        
        # Add some default control table entries
        self.populate_control_table()
        
    def setup_programming_tab(self):
        # Graph display area (simplified placeholder)
        graph_frame = ttk.LabelFrame(self.prog_tab, text="Motor Feedback", padding="5")
        graph_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Canvas for simple graph
        self.canvas = tk.Canvas(graph_frame, bg="white", height=300)
        self.canvas.pack(fill="both", expand=True)
        
        # Checkboxes for data selection
        check_frame = ttk.Frame(graph_frame)
        check_frame.pack(fill="x", pady=5)
        
        self.plot_vars = {}
        for name in ['Position', 'Torque', 'Speed', 'Current', 'Temperature', 'Voltage']:
            var = tk.BooleanVar(value=(name in ['Position', 'Torque']))
            self.plot_vars[name.lower()] = var
            ttk.Checkbutton(check_frame, text=name, variable=var).pack(side="left", padx=5)
        
        # Control frame
        control_frame = ttk.LabelFrame(self.prog_tab, text="Servo Control", padding="10")
        control_frame.pack(fill="x", padx=5, pady=5)
        
        # Control type selection
        ttk.Radiobutton(control_frame, text="Write", value="write").grid(row=0, column=0)
        ttk.Radiobutton(control_frame, text="Sync Write", value="sync").grid(row=0, column=1)
        ttk.Radiobutton(control_frame, text="Reg Write", value="reg").grid(row=0, column=2)
        
        self.torque_enable = tk.BooleanVar()
        ttk.Checkbutton(control_frame, text="Torque Enable", variable=self.torque_enable, 
                       command=self.toggle_torque).grid(row=0, column=3)
        
        # Position control slider
        ttk.Label(control_frame, text="Position:").grid(row=1, column=0)
        self.position_slider = ttk.Scale(control_frame, from_=0, to=4095, orient="horizontal", length=300)
        self.position_slider.grid(row=1, column=1, columnspan=3, padx=5)
        
        ttk.Label(control_frame, text="Goal:").grid(row=2, column=0)
        self.goal_entry = ttk.Entry(control_frame, width=10)
        self.goal_entry.grid(row=2, column=1)
        self.goal_entry.insert(0, "2048")
        
        ttk.Button(control_frame, text="Action", command=self.send_position).grid(row=2, column=2, padx=5)
        
        # Feedback display
        feedback_frame = ttk.LabelFrame(self.prog_tab, text="Servo Feedback", padding="10")
        feedback_frame.pack(fill="x", padx=5, pady=5)
        
        self.feedback_labels = {}
        feedback_items = [
            ('Voltage', '0.0V'), ('Torque', '0'),
            ('Current', '0'), ('Speed', '0'),
            ('Temperature', '0'), ('Position', '0'),
            ('Moving', '0'), ('Goal', '0')
        ]
        
        for i, (name, default) in enumerate(feedback_items):
            row = i // 2
            col = (i % 2) * 2
            ttk.Label(feedback_frame, text=f"{name}:").grid(row=row, column=col, sticky="w", padx=5)
            label = ttk.Label(feedback_frame, text=default)
            label.grid(row=row, column=col+1, sticky="w", padx=5)
            self.feedback_labels[name.lower()] = label
        
        # Auto debug controls
        auto_frame = ttk.LabelFrame(self.prog_tab, text="Auto Debug", padding="10")
        auto_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(auto_frame, text="Start:").grid(row=0, column=0)
        self.start_pos = ttk.Entry(auto_frame, width=10)
        self.start_pos.grid(row=0, column=1)
        self.start_pos.insert(0, "0")
        
        ttk.Label(auto_frame, text="End:").grid(row=0, column=2)
        self.end_pos = ttk.Entry(auto_frame, width=10)
        self.end_pos.grid(row=0, column=3)
        self.end_pos.insert(0, "4095")
        
        ttk.Label(auto_frame, text="Delay(ms):").grid(row=1, column=0)
        self.sweep_delay = ttk.Entry(auto_frame, width=10)
        self.sweep_delay.grid(row=1, column=1)
        self.sweep_delay.insert(0, "2500")
        
        ttk.Button(auto_frame, text="Sweep", command=self.start_sweep).grid(row=1, column=2, padx=5)
        ttk.Button(auto_frame, text="Stop", command=self.stop_sweep).grid(row=1, column=3, padx=5)
        
    def refresh_com_ports(self):
        """Refresh the list of available COM ports."""
        ports = self.get_usb_serial_ports()
        self.com_port['values'] = ports
        
        # If current selection is not in the list, select the first port
        if self.com_port.get() not in ports:
            if ports:
                self.com_port.set(ports[0])
            else:
                self.com_port.set("")
    
    def populate_control_table(self):
        # Default control table entries based on the images
        default_entries = [
            (0, "Firmware Main Version NO.", 3, "EPROM", "R"),
            (1, "Firmware Secondary Version", 9, "EPROM", "R"),
            (3, "Servo Main Version", 9, "EPROM", "R"),
            (4, "Servo Sub Version", 5, "EPROM", "R"),
            (5, "ID", 2, "EPROM", "R/W"),
            (6, "Baud Rate", 0, "EPROM", "R/W"),
            (7, "Return Delay Time", 0, "EPROM", "R/W"),
            (8, "Status Return Level", 1, "EPROM", "R/W"),
            (9, "Min Position Limit", 0, "EPROM", "R/W"),
            (11, "Max Position Limit", 4095, "EPROM", "R/W"),
            (13, "Max Temperature Limit", 80, "EPROM", "R/W"),
            (14, "Max Input Voltage", 100, "EPROM", "R/W"),
            (15, "Min Input Voltage", 55, "EPROM", "R/W"),
            (16, "Max Torque Limit", 1000, "EPROM", "R/W"),
            (18, "Setting Byte", 12, "EPROM", "R/W"),
            (19, "Protection Switch", 44, "EPROM", "R/W"),
            (20, "LED Alarm Condition", 47, "EPROM", "R/W"),
            (21, "Position P Gain", 32, "EPROM", "R/W"),
        ]
        
        for entry in default_entries:
            self.control_table.insert("", "end", values=entry)
    
    def toggle_connection(self):
        if not self.connected:
            self.connect_motor_bus()
        else:
            self.disconnect_motor_bus()
    
    def connect_motor_bus(self):
        try:
            port = self.com_port.get()
            baudrate = int(self.baudrate.get())
            
            if not port:
                messagebox.showerror("Error", "Please select a COM port")
                return
            
            # Initialize motor bus with empty motors dict and connect without handshake
            # This allows us to connect without knowing which motors are present
            self.motor_bus = FeetechMotorsBus(port=port, motors={})
            self.motor_bus.connect(handshake=False)  # Skip handshake for discovery
            
            # Set the baudrate
            if hasattr(self.motor_bus.port_handler, 'setBaudRate'):
                self.motor_bus.port_handler.setBaudRate(baudrate)
            
            self.connected = True
            self.connect_btn.config(text="Close")
            messagebox.showinfo("Success", f"Connected to motor bus on {port} at {baudrate} baud")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to connect: {str(e)}")
            self.motor_bus = None
            self.connected = False
    
    def disconnect_motor_bus(self):
        try:
            if self.motor_bus is not None:
                self.motor_bus.disconnect()
                self.motor_bus = None
            self.connected = False
            self.connect_btn.config(text="Open")
            messagebox.showinfo("Success", "Disconnected from motor bus")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to disconnect: {str(e)}")
            self.motor_bus = None
            self.connected = False
    
    def search_motors(self):
        if not self.connected or self.motor_bus is None:
            messagebox.showwarning("Warning", "Please connect to motor bus first")
            return
        
        # Clear existing entries
        for item in self.motor_tree.get_children():
            self.motor_tree.delete(item)
        
        try:
            # Use broadcast_ping to discover all connected motors efficiently
            self.search_btn.config(text="Searching...", state="disabled")
            self.root.update()
            
            # Broadcast ping returns dict of {motor_id: model_number}
            motors_found = self.motor_bus.broadcast_ping()
            
            if motors_found:
                # Map model numbers to model names (common Feetech models)
                model_map = {
                    3020: "STS3020",
                    3215: "STS3215", 
                    3032: "STS3032",
                    1000: "SCS15",
                    15: "SCS15",
                    3038: "STS3038",
                    # Add more model mappings as needed
                }
                
                for motor_id, model_num in motors_found.items():
                    model_name = model_map.get(model_num, f"Model_{model_num}")
                    self.motor_tree.insert("", "end", values=(motor_id, model_name))
                
                self.root.update()
                messagebox.showinfo("Search Complete", f"Found {len(motors_found)} motor(s)")
            else:
                messagebox.showwarning("Search Complete", "No motors found")
                
        except Exception as e:
            messagebox.showerror("Search Error", f"Error during motor search: {str(e)}")
        finally:
            self.search_btn.config(text="Search", state="normal")
    
    def on_motor_select(self, event):
        selection = self.motor_tree.selection()
        if selection:
            item = self.motor_tree.item(selection[0])
            self.selected_motor_id = item['values'][0]
            self.edit_id.delete(0, tk.END)
            self.edit_id.insert(0, str(self.selected_motor_id))
            self.update_control_table()
    
    def update_control_table(self):
        if not self.connected or not self.selected_motor_id:
            return
        
        # In real implementation, read actual values from motor
        # For now, just highlight the selected motor's row
        for item in self.control_table.get_children():
            values = self.control_table.item(item)['values']
            if values[0] == 5:  # ID row
                self.control_table.set(item, 'Value', self.selected_motor_id)
    
    def set_value(self):
        if not self.connected:
            messagebox.showwarning("Warning", "Please connect to motor bus first")
            return
        
        selection = self.control_table.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a register to modify")
            return
        
        try:
            value = int(self.edit_value.get())
            # In real implementation, write to motor using motor_bus.write()
            messagebox.showinfo("Success", f"Value {value} written successfully")
        except ValueError:
            messagebox.showerror("Error", "Invalid value")
    
    def toggle_torque(self):
        if not self.connected or not self.selected_motor_id:
            return
        
        if self.torque_enable.get():
            # self.motor_bus.enable_torque([self.selected_motor_id])
            print(f"Torque enabled for motor {self.selected_motor_id}")
        else:
            # self.motor_bus.disable_torque([self.selected_motor_id])
            print(f"Torque disabled for motor {self.selected_motor_id}")
    
    def send_position(self):
        if not self.connected or not self.selected_motor_id:
            messagebox.showwarning("Warning", "Please connect and select a motor")
            return
        
        try:
            goal = int(self.goal_entry.get())
            # self.motor_bus.write("Goal_Position", self.selected_motor_id, goal)
            self.position_slider.set(goal)
            print(f"Sent position {goal} to motor {self.selected_motor_id}")
        except ValueError:
            messagebox.showerror("Error", "Invalid position value")
    
    def start_sweep(self):
        if not self.connected or not self.selected_motor_id:
            messagebox.showwarning("Warning", "Please connect and select a motor")
            return
        
        # Start sweep in a separate thread
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.sweep_motor)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
    
    def stop_sweep(self):
        self.monitoring = False
    
    def sweep_motor(self):
        try:
            start = int(self.start_pos.get())
            end = int(self.end_pos.get())
            delay = int(self.sweep_delay.get()) / 1000.0
            
            while self.monitoring:
                # Sweep from start to end
                for pos in range(start, end, 50):
                    if not self.monitoring:
                        break
                    # self.motor_bus.write("Goal_Position", self.selected_motor_id, pos)
                    self.root.after(0, self.position_slider.set, pos)
                    time.sleep(delay / ((end - start) / 50))
                
                # Sweep back
                for pos in range(end, start, -50):
                    if not self.monitoring:
                        break
                    # self.motor_bus.write("Goal_Position", self.selected_motor_id, pos)
                    self.root.after(0, self.position_slider.set, pos)
                    time.sleep(delay / ((end - start) / 50))
        except Exception as e:
            print(f"Sweep error: {e}")
    
    def save_config(self):
        messagebox.showinfo("Info", "Save configuration - to be implemented")
    
    def load_config(self):
        messagebox.showinfo("Info", "Load configuration - to be implemented")
    
    def go_online(self):
        messagebox.showinfo("Info", "Go online - to be implemented")
    
    def recovery(self):
        if messagebox.askyesno("Recovery", "Are you sure you want to restore factory settings?"):
            # self.motor_bus.reset_calibration()
            messagebox.showinfo("Success", "Factory settings restored")


def main():
    root = tk.Tk()
    app = ServoDebugUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
