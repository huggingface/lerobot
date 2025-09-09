#!/usr/bin/env python3
"""
Servo Motor Debug UI for LeRobot Motors Bus
A tkinter-based debugging interface for servo motors using the lerobot motors_bus API
"""

import glob
import platform
import threading
import time
import tkinter as tk
from collections import deque
from tkinter import messagebox, ttk
from typing import List

from lerobot.motors.feetech import FeetechMotorsBus
from lerobot.motors.motors_bus import Motor, MotorNormMode


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

        # Plot settings
        self.plot_colors = {
            'position': '#0066CC',
            'torque': '#FF6600',
            'speed': '#009900',
            'current': '#CC0066',
            'temperature': '#FF0000',
            'voltage': '#9900CC'
        }

        self.setup_ui()

    def get_motor_name_from_id(self, motor_id: int) -> str | None:
        """Get the motor name key from motor ID."""
        if not hasattr(self, 'discovered_motors') or not self.discovered_motors:
            return None

        for name, motor in self.discovered_motors.items():
            if motor.id == motor_id:
                return name
        return None

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

        # Programming Tab (control table)
        self.debug_tab = ttk.Frame(right_panel)
        right_panel.add(self.debug_tab, text="Programming")
        self.setup_debug_tab()

        # Debug Tab (graphs and monitoring)
        self.prog_tab = ttk.Frame(right_panel)
        right_panel.add(self.prog_tab, text="Debug")
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

        # Bind double-click for direct editing
        self.control_table.bind("<Double-1>", self.on_table_double_click)

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
        # Control table entries mapped to actual Feetech registers
        self.control_registers = [
            # (address, name, register_key, area, access)
            (0, "Firmware Major Version", "Firmware_Major_Version", "EPROM", "R"),
            (1, "Firmware Minor Version", "Firmware_Minor_Version", "EPROM", "R"),
            (3, "Model Number", "Model_Number", "EPROM", "R"),
            (5, "ID", "ID", "EPROM", "R/W"),
            (6, "Baud Rate", "Baud_Rate", "EPROM", "R/W"),
            (7, "Return Delay Time", "Return_Delay_Time", "EPROM", "R/W"),
            (8, "Response Status Level", "Response_Status_Level", "EPROM", "R/W"),
            (9, "Min Position Limit", "Min_Position_Limit", "EPROM", "R/W"),
            (11, "Max Position Limit", "Max_Position_Limit", "EPROM", "R/W"),
            (13, "Max Temperature Limit", "Max_Temperature_Limit", "EPROM", "R/W"),
            (14, "Max Voltage Limit", "Max_Voltage_Limit", "EPROM", "R/W"),
            (15, "Min Voltage Limit", "Min_Voltage_Limit", "EPROM", "R/W"),
            (16, "Max Torque Limit", "Max_Torque_Limit", "EPROM", "R/W"),
            (40, "Torque Enable", "Torque_Enable", "SRAM", "R/W"),
            (42, "Goal Position", "Goal_Position", "SRAM", "R/W"),
            (46, "Goal Velocity", "Goal_Velocity", "SRAM", "R/W"),
            (48, "Torque Limit", "Torque_Limit", "SRAM", "R/W"),
            (56, "Present Position", "Present_Position", "SRAM", "R"),
            (58, "Present Velocity", "Present_Velocity", "SRAM", "R"),
            (60, "Present Load", "Present_Load", "SRAM", "R"),
            (62, "Present Voltage", "Present_Voltage", "SRAM", "R"),
            (63, "Present Temperature", "Present_Temperature", "SRAM", "R"),
            (66, "Moving", "Moving", "SRAM", "R"),
        ]

        # Initially populate with default values
        for addr, name, reg_key, area, access in self.control_registers:
            self.control_table.insert("", "end", values=(addr, name, "---", area, access))

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

            # First, try broadcast_ping with the empty motor bus
            print("Debug: About to call broadcast_ping()")
            motors_found = self.motor_bus.broadcast_ping()
            print(f"Debug: broadcast_ping() returned: {motors_found}")

            # If broadcast_ping returns None, try with retry
            if motors_found is None:
                print("Debug: broadcast_ping() returned None, trying with retry...")
                motors_found = self.motor_bus.broadcast_ping(num_retry=3)
                print(f"Debug: broadcast_ping() with retry returned: {motors_found}")

            if motors_found:
                # Map model numbers to model names (common Feetech models)
                model_map = {
                    777: "sts3215",   # Your motors are STS3215 with model number 777
                    # Add more model mappings as needed
                }

                # Store discovered motors for later use
                self.discovered_motors = {}
                print(f"Debug: Processing {len(motors_found)} motors")
                for motor_id, model_num in motors_found.items():
                    try:
                        print(f"Debug: Processing motor {motor_id} with model {model_num}")
                        model_name = model_map.get(model_num, "sts3215")  # Default to sts3020
                        display_name = model_map.get(model_num, f"Model_{model_num}").upper()
                        print(f"Debug: Mapped to model_name='{model_name}', display_name='{display_name}'")

                        # Add to tree view
                        self.motor_tree.insert("", "end", values=(motor_id, display_name))
                        print(f"Debug: Added to tree view: {motor_id}, {display_name}")

                        # Store motor info for reading/writing
                        motor_key = f"motor_{motor_id}"
                        print(f"Debug: About to create Motor for {motor_key} with model='{model_name}'")
                        try:
                            motor_obj = Motor(
                                id=motor_id,
                                model=model_name,
                                norm_mode=MotorNormMode.DEGREES  # Use degrees mode for debug interface
                            )
                            self.discovered_motors[motor_key] = motor_obj
                            print(f"Debug: Successfully created and stored motor {motor_key}")
                            print(f"Debug: discovered_motors now has {len(self.discovered_motors)} motors: {list(self.discovered_motors.keys())}")
                        except Exception as motor_creation_error:
                            print(f"Debug: Error creating Motor object for {motor_id}: {motor_creation_error}")
                            print(f"Debug: Motor args: id={motor_id}, model='{model_name}', norm_mode=MotorNormMode.NONE")
                            print(f"Debug: Available MotorNormMode values: {list(MotorNormMode)}")
                            # Still continue with next motor
                    except Exception as motor_error:
                        print(f"Debug: Error processing motor {motor_id}: {motor_error}")
                        import traceback
                        traceback.print_exc()
                        # Continue with next motor instead of failing completely

                print("Debug: Finished processing all motors")

                # Recreate motor bus with discovered motors for proper read/write operations
                try:
                    port = self.com_port.get()
                    baudrate = int(self.baudrate.get())

                    # Disconnect current bus
                    if self.motor_bus:
                        self.motor_bus.disconnect()

                    # Create new bus with discovered motors
                    print(f"Debug: Creating new motor bus with {len(self.discovered_motors)} motors")
                    print(f"Debug: discovered_motors contents: {list(self.discovered_motors.keys())}")
                    self.motor_bus = FeetechMotorsBus(port=port, motors=self.discovered_motors)
                    self.motor_bus.connect(handshake=False)

                    # Set baudrate
                    if hasattr(self.motor_bus.port_handler, 'setBaudRate'):
                        self.motor_bus.port_handler.setBaudRate(baudrate)

                    print("Debug: Successfully recreated motor bus with discovered motors")
                    print(f"Debug: Motor bus now has motors: {list(self.motor_bus.motors.keys()) if hasattr(self.motor_bus, 'motors') else 'No motors attr'}")
                except Exception as e:
                    print(f"Debug: Failed to recreate motor bus: {e}")

                self.root.update()
                messagebox.showinfo("Search Complete", f"Found {len(motors_found)} motor(s)")
            else:
                print("Debug: motors_found is falsy (None or empty)")
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

            # Start feedback monitoring if not already running
            if not hasattr(self, '_feedback_running'):
                print(f"Debug: Starting feedback monitoring for motor ID {self.selected_motor_id}")
                self._feedback_running = True
                self.start_feedback_monitoring()
            else:
                print(f"Debug: Feedback monitoring already running, selected motor ID {self.selected_motor_id}")

    def update_control_table(self):
        if not self.connected or self.motor_bus is None or not self.selected_motor_id:
            return

        # Get motor name for selected motor
        motor_name = self.get_motor_name_from_id(self.selected_motor_id)
        if not motor_name:
            print(f"Could not find motor name for ID {self.selected_motor_id}")
            # Show placeholder if no motor found
            items = list(self.control_table.get_children())
            for i in range(len(items)):
                self.control_table.set(items[i], 'Value', "---")
            return

        try:
            # Read actual values from the selected motor
            items = list(self.control_table.get_children())

            for i, (addr, name, reg_key, area, access) in enumerate(self.control_registers):
                try:
                    # Read the register value using motor name
                    value = self.motor_bus.read(reg_key, motor_name, normalize=False)
                    # Update the control table display
                    if i < len(items):
                        self.control_table.set(items[i], 'Value', str(value))
                except Exception:
                    # If read fails, show error indicator
                    if i < len(items):
                        self.control_table.set(items[i], 'Value', "ERR")

        except Exception as e:
            print(f"Error updating control table: {e}")
            # Show ERR for all items if something goes wrong
            items = list(self.control_table.get_children())
            for i in range(len(items)):
                self.control_table.set(items[i], 'Value', "ERR")

    def on_table_double_click(self, event):
        """Handle double-click on control table for direct editing."""
        if not self.connected or self.motor_bus is None or not self.selected_motor_id:
            messagebox.showwarning("Warning", "Please connect and select a motor first")
            return

        # Get the selected item and column
        item_id = self.control_table.selection()[0] if self.control_table.selection() else None
        if not item_id:
            return

        # Get the column that was clicked
        column = self.control_table.identify_column(event.x)
        if column != '#3':  # Only allow editing the 'Value' column (column 3)
            return

        # Get the register info
        item_index = self.control_table.index(item_id)
        if item_index >= len(self.control_registers):
            return

        addr, name, reg_key, area, access = self.control_registers[item_index]

        # Check if register is writable
        if 'W' not in access:
            messagebox.showinfo("Info", f"Register '{name}' is read-only")
            return

        # Get current value
        current_value = self.control_table.item(item_id)['values'][2]  # Value column

        # Create a simple input dialog
        new_value = tk.simpledialog.askstring(
            "Edit Value",
            f"Edit {name}:\nAddress: {addr}\nCurrent value: {current_value}",
            initialvalue=str(current_value)
        )

        if new_value is not None:
            try:
                # Convert to integer
                int_value = int(new_value)

                # Write to motor
                motor_name = self.get_motor_name_from_id(self.selected_motor_id)
                if motor_name:
                    self.motor_bus.write(reg_key, motor_name, int_value, normalize=False)
                    messagebox.showinfo("Success", f"Value {int_value} written to {name}")

                    # Update the control table
                    self.update_control_table()
                else:
                    messagebox.showerror("Error", f"Motor ID {self.selected_motor_id} not found")

            except ValueError:
                messagebox.showerror("Error", "Invalid value - must be an integer")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to write value: {str(e)}")

    def set_value(self):
        if not self.connected or self.motor_bus is None:
            messagebox.showwarning("Warning", "Please connect to motor bus first")
            return

        if not self.selected_motor_id:
            messagebox.showwarning("Warning", "Please select a motor first")
            return

        selection = self.control_table.selection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a register to modify")
            return

        try:
            value = int(self.edit_value.get())
            motor_id = int(self.edit_id.get())

            # Get motor name
            motor_name = self.get_motor_name_from_id(motor_id)
            if not motor_name:
                messagebox.showerror("Error", f"Motor ID {motor_id} not found")
                return

            # Get the selected register info
            item_index = self.control_table.index(selection[0])
            if item_index < len(self.control_registers):
                addr, name, reg_key, area, access = self.control_registers[item_index]

                # Check if register is writable
                if 'W' not in access:
                    messagebox.showwarning("Warning", f"Register '{name}' is read-only")
                    return

                # Write to motor using motor name
                self.motor_bus.write(reg_key, motor_name, value, normalize=False)
                messagebox.showinfo("Success", f"Value {value} written to {name} on motor {motor_id}")

                # Refresh the control table to show the updated value
                self.update_control_table()
            else:
                messagebox.showerror("Error", "Invalid register selection")

        except ValueError:
            messagebox.showerror("Error", "Invalid value - must be an integer")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to write value: {str(e)}")

    def toggle_torque(self):
        if not self.connected or self.motor_bus is None or not self.selected_motor_id:
            return

        motor_name = self.get_motor_name_from_id(self.selected_motor_id)
        if not motor_name:
            return

        try:
            if self.torque_enable.get():
                self.motor_bus.write("Torque_Enable", motor_name, 1, normalize=False)
                print(f"Torque enabled for motor {self.selected_motor_id}")
            else:
                self.motor_bus.write("Torque_Enable", motor_name, 0, normalize=False)
                print(f"Torque disabled for motor {self.selected_motor_id}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to toggle torque: {str(e)}")

    def send_position(self):
        if not self.connected or self.motor_bus is None or not self.selected_motor_id:
            messagebox.showwarning("Warning", "Please connect and select a motor")
            return

        motor_name = self.get_motor_name_from_id(self.selected_motor_id)
        if not motor_name:
            messagebox.showerror("Error", f"Motor ID {self.selected_motor_id} not found")
            return

        try:
            goal = int(self.goal_entry.get())
            self.motor_bus.write("Goal_Position", motor_name, goal, normalize=False)
            self.position_slider.set(goal)
            print(f"Sent position {goal} to motor {self.selected_motor_id}")
        except ValueError:
            messagebox.showerror("Error", "Invalid position value")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to send position: {str(e)}")

    def start_sweep(self):
        if not self.connected or self.motor_bus is None or not self.selected_motor_id:
            messagebox.showwarning("Warning", "Please connect and select a motor")
            return

        # Start sweep in a separate thread
        if not self.monitoring:
            self.monitoring = True
            self.monitor_thread = threading.Thread(target=self.sweep_motor)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

            # Also start feedback monitoring
            self.start_feedback_monitoring()

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
                    try:
                        motor_name = self.get_motor_name_from_id(self.selected_motor_id)
                        if motor_name:
                            self.motor_bus.write("Goal_Position", motor_name, pos, normalize=False)
                            self.root.after(0, self.position_slider.set, pos)
                    except Exception as e:
                        print(f"Sweep write error: {e}")
                    time.sleep(delay / max(1, ((end - start) / 50)))

                # Sweep back
                for pos in range(end, start, -50):
                    if not self.monitoring:
                        break
                    try:
                        motor_name = self.get_motor_name_from_id(self.selected_motor_id)
                        if motor_name:
                            self.motor_bus.write("Goal_Position", motor_name, pos, normalize=False)
                            self.root.after(0, self.position_slider.set, pos)
                    except Exception as e:
                        print(f"Sweep write error: {e}")
                    time.sleep(delay / max(1, ((end - start) / 50)))
        except Exception as e:
            print(f"Sweep error: {e}")

    def start_feedback_monitoring(self):
        """Start periodic feedback updates for the programming tab."""
        print("Debug: start_feedback_monitoring called")
        self.update_feedback()

    def update_feedback(self):
        """Update the feedback display with current motor values."""
        if not self.connected or self.motor_bus is None or not self.selected_motor_id:
            # Schedule next update anyway
            self.root.after(200, self.update_feedback)  # Update every 200ms
            return

        motor_name = self.get_motor_name_from_id(self.selected_motor_id)
        if not motor_name:
            print(f"Debug: No motor name found for feedback update of ID {self.selected_motor_id}")
            # Schedule next update anyway
            self.root.after(200, self.update_feedback)
            return

        print(f"Debug: Updating feedback for motor {motor_name} (ID {self.selected_motor_id})")
        print(f"Debug: Available feedback labels: {list(self.feedback_labels.keys())}")

        try:
            # Read feedback values from motor
            feedback_registers = {
                'voltage': 'Present_Voltage',
                'torque': 'Present_Load',
                'current': 'Present_Load',  # Same as torque for Feetech
                'speed': 'Present_Velocity',
                'temperature': 'Present_Temperature',
                'position': 'Present_Position',
                'moving': 'Moving',
                'goal': 'Goal_Position'
            }

            for display_name, reg_key in feedback_registers.items():
                try:
                    print(f"Debug: Reading {reg_key} for {display_name}")
                    value = self.motor_bus.read(reg_key, motor_name, normalize=False)
                    print(f"Debug: Got value {value} for {display_name}")

                    # Format the value appropriately
                    if display_name == 'voltage':
                        formatted_value = f"{value/10:.1f}V"  # Voltage is in 0.1V units
                    elif display_name == 'temperature':
                        formatted_value = f"{value}°C"
                    elif display_name == 'moving':
                        formatted_value = "Yes" if value else "No"
                    else:
                        formatted_value = str(value)

                    print(f"Debug: Formatted value: {formatted_value}")

                    # Update the label
                    if display_name in self.feedback_labels:
                        self.feedback_labels[display_name].config(text=formatted_value)
                        print(f"Debug: Updated label for {display_name}")
                    else:
                        print(f"Debug: No label found for {display_name}")

                    # Store data for plotting (use raw values for plotting)
                    if display_name in self.plot_data:
                        self.plot_data[display_name].append(value)
                        print(f"Debug: Added {value} to plot_data[{display_name}]")

                except Exception as e:
                    print(f"Debug: Error reading {reg_key} for {display_name}: {e}")
                    # If individual read fails, show error
                    if display_name in self.feedback_labels:
                        self.feedback_labels[display_name].config(text="ERR")

            # Update the plot after all data is collected
            self.update_plot()

        except Exception as e:
            print(f"Feedback update error: {e}")

        # Schedule next update
        self.root.after(200, self.update_feedback)  # Update every 200ms

    def update_plot(self):
        """Update the canvas plot with current data."""
        if not hasattr(self, 'canvas') or not self.canvas:
            return

        # Clear the canvas
        self.canvas.delete("all")

        # Get canvas dimensions
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width <= 1 or canvas_height <= 1:
            # Canvas not ready yet
            return

        # Plot margins
        margin_left = 50
        margin_right = 20
        margin_top = 20
        margin_bottom = 50

        plot_width = canvas_width - margin_left - margin_right
        plot_height = canvas_height - margin_top - margin_bottom

        if plot_width <= 0 or plot_height <= 0:
            return

        # Draw axes
        # X-axis (time)
        self.canvas.create_line(margin_left, canvas_height - margin_bottom,
                               canvas_width - margin_right, canvas_height - margin_bottom,
                               fill="black", width=2)

        # Y-axis (values)
        self.canvas.create_line(margin_left, margin_top,
                               margin_left, canvas_height - margin_bottom,
                               fill="black", width=2)

        # Plot selected data series
        for data_name, var in self.plot_vars.items():
            if not var.get() or data_name not in self.plot_data:
                continue

            data = list(self.plot_data[data_name])
            if len(data) < 2:
                continue

            # Normalize data for plotting
            if len(data) > 0:
                min_val = min(data)
                max_val = max(data)
                val_range = max_val - min_val if max_val != min_val else 1

                # Create points
                points = []
                for i, value in enumerate(data):
                    x = margin_left + (i / max(1, len(data) - 1)) * plot_width
                    y = canvas_height - margin_bottom - ((value - min_val) / val_range) * plot_height
                    points.extend([x, y])

                # Draw the line
                if len(points) >= 4:
                    color = self.plot_colors.get(data_name, "#000000")
                    self.canvas.create_line(points, fill=color, width=2)

                    # Add legend
                    legend_y = margin_top + list(self.plot_vars.keys()).index(data_name) * 20
                    self.canvas.create_line(canvas_width - margin_right - 30, legend_y,
                                          canvas_width - margin_right - 10, legend_y,
                                          fill=color, width=3)
                    self.canvas.create_text(canvas_width - margin_right - 35, legend_y,
                                          text=data_name.capitalize(), anchor="e", fill=color)

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
