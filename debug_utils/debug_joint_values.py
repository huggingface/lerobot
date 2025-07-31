#!/usr/bin/env python

"""
Debug script to log joint values from both leader and follower to CSV
This will help analyze the wrist_flex (ID=4) servo issues
"""

import csv
import signal
import time
from datetime import datetime
from pathlib import Path

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech.feetech import FeetechMotorsBus


class JointLogger:
    def __init__(self, leader_port, follower_port):
        self.leader_port = leader_port
        self.follower_port = follower_port
        self.running = True

        # Create CSV file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_file = Path(f"joint_debug_{timestamp}.csv")

        # Motor configurations
        self.leader_motors = {
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }

        self.follower_motors = {
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }

        self.leader_bus = None
        self.follower_bus = None

        # Setup signal handler for clean exit
        signal.signal(signal.SIGINT, self.signal_handler)

    def signal_handler(self, sig, frame):
        print("\n🛑 Stopping data collection...")
        self.running = False

    def connect_buses(self):
        """Connect to both leader and follower buses"""
        print("🔌 Connecting to leader arm...")
        self.leader_bus = FeetechMotorsBus(port=self.leader_port, motors=self.leader_motors)

        try:
            self.leader_bus.connect()
            print("✅ Leader connected")
        except Exception as e:
            print(f"❌ Leader connection failed: {e}")
            return False

        print("🔌 Connecting to follower arm...")
        self.follower_bus = FeetechMotorsBus(port=self.follower_port, motors=self.follower_motors)

        try:
            self.follower_bus.connect()
            print("✅ Follower connected")
        except Exception as e:
            print(f"❌ Follower connection failed: {e}")
            return False

        return True

    def read_joint_data(self, bus, prefix):
        """Read all joint data from a bus"""
        data = {}
        timestamp = time.time()

        try:
            # Read positions
            positions = bus.sync_read("Present_Position", normalize=False)
            for joint, pos in positions.items():
                data[f"{prefix}_{joint}_pos"] = pos

            # Read loads for health monitoring
            loads = bus.sync_read("Present_Load", normalize=False)
            for joint, load in loads.items():
                data[f"{prefix}_{joint}_load"] = load

            # Read temperatures (especially important for wrist_flex)
            temps = bus.sync_read("Present_Temperature", normalize=False)
            for joint, temp in temps.items():
                data[f"{prefix}_{joint}_temp"] = temp

            # Check torque enable status
            for joint in positions.keys():
                try:
                    torque = bus.read("Torque_Enable", joint)
                    data[f"{prefix}_{joint}_torque_enabled"] = torque
                except:
                    data[f"{prefix}_{joint}_torque_enabled"] = -1

        except Exception as e:
            print(f"❌ Error reading from {prefix}: {e}")
            # Fill with error values
            for joint in self.leader_motors.keys():
                data[f"{prefix}_{joint}_pos"] = -9999
                data[f"{prefix}_{joint}_load"] = -9999
                data[f"{prefix}_{joint}_temp"] = -9999
                data[f"{prefix}_{joint}_torque_enabled"] = -9999

        data["timestamp"] = timestamp
        return data

    def log_data(self):
        """Main logging loop"""

        # Create CSV headers
        headers = ["timestamp", "elapsed_time"]

        # Add all joint data columns
        for joint in self.leader_motors.keys():
            for prefix in ["leader", "follower"]:
                headers.extend(
                    [
                        f"{prefix}_{joint}_pos",
                        f"{prefix}_{joint}_load",
                        f"{prefix}_{joint}_temp",
                        f"{prefix}_{joint}_torque_enabled",
                    ]
                )

        # Add difference columns for positions
        for joint in self.leader_motors.keys():
            headers.append(f"diff_{joint}_pos")

        print(f"📝 Logging to: {self.csv_file}")
        print("📊 Columns:", len(headers))

        start_time = time.time()

        with open(self.csv_file, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers)
            writer.writeheader()

            sample_count = 0
            last_print_time = time.time()

            while self.running:
                try:
                    current_time = time.time()

                    # Read data from both arms
                    leader_data = self.read_joint_data(self.leader_bus, "leader")
                    follower_data = self.read_joint_data(self.follower_bus, "follower")

                    # Combine data
                    row_data = {"timestamp": current_time, "elapsed_time": current_time - start_time}

                    row_data.update(leader_data)
                    row_data.update(follower_data)

                    # Calculate position differences (leader - follower)
                    for joint in self.leader_motors.keys():
                        leader_pos = leader_data.get(f"leader_{joint}_pos", 0)
                        follower_pos = follower_data.get(f"follower_{joint}_pos", 0)
                        if leader_pos != -9999 and follower_pos != -9999:
                            row_data[f"diff_{joint}_pos"] = leader_pos - follower_pos
                        else:
                            row_data[f"diff_{joint}_pos"] = -9999

                    # Write to CSV
                    writer.writerow(row_data)
                    csvfile.flush()  # Ensure data is written immediately

                    sample_count += 1

                    # Print status every 2 seconds
                    if current_time - last_print_time >= 2.0:
                        print(f"📈 Samples: {sample_count}, Time: {row_data['elapsed_time']:.1f}s")

                        # Print current wrist_flex values (the problematic joint)
                        wrist_leader = leader_data.get("leader_wrist_flex_pos", "N/A")
                        wrist_follower = follower_data.get("follower_wrist_flex_pos", "N/A")
                        wrist_diff = row_data.get("diff_wrist_flex_pos", "N/A")
                        wrist_load = follower_data.get("follower_wrist_flex_load", "N/A")
                        wrist_temp = follower_data.get("follower_wrist_flex_temp", "N/A")

                        print(f"   🎯 Wrist_Flex: L={wrist_leader}, F={wrist_follower}, Diff={wrist_diff}")
                        print(f"      Load={wrist_load}, Temp={wrist_temp}°C")

                        # Check for concerning values
                        if isinstance(wrist_load, (int, float)) and wrist_load > 800:
                            print(f"   ⚠️  HIGH LOAD detected on wrist_flex: {wrist_load}")
                        if isinstance(wrist_temp, (int, float)) and wrist_temp > 55:
                            print(f"   ⚠️  HIGH TEMP detected on wrist_flex: {wrist_temp}°C")
                        if isinstance(wrist_diff, (int, float)) and abs(wrist_diff) > 500:
                            print(f"   ⚠️  LARGE POSITION DIFF detected: {wrist_diff}")

                        last_print_time = current_time

                    # Small delay to avoid overwhelming the servos
                    time.sleep(0.05)  # 20 Hz sampling

                except Exception as e:
                    print(f"❌ Error in logging loop: {e}")
                    time.sleep(0.1)

        print(f"✅ Data collection complete. {sample_count} samples saved to {self.csv_file}")

    def disconnect(self):
        """Clean disconnect from both buses"""
        if self.leader_bus:
            try:
                self.leader_bus.disconnect()
                print("🔌 Leader disconnected")
            except:
                pass

        if self.follower_bus:
            try:
                self.follower_bus.disconnect()
                print("🔌 Follower disconnected")
            except:
                pass

    def run(self):
        """Main execution function"""
        print("🚀 Joint Value Debug Logger")
        print("=" * 50)

        if not self.connect_buses():
            print("❌ Failed to connect to arms")
            return

        try:
            print("\n🎯 Starting data collection...")
            print("💡 Press Ctrl+C to stop logging")
            print("💡 Move the leader arm to test teleoperation")

            self.log_data()

        finally:
            self.disconnect()


def analyze_csv(csv_file):
    """Quick analysis of the logged data"""
    import pandas as pd

    try:
        df = pd.read_csv(csv_file)
        print(f"\n📊 Analysis of {csv_file}")
        print("=" * 50)

        print(f"Total samples: {len(df)}")
        print(f"Duration: {df['elapsed_time'].max():.2f} seconds")

        # Focus on wrist_flex (the problematic joint)
        wrist_cols = [col for col in df.columns if "wrist_flex" in col]
        print("\nWrist Flex Analysis:")
        for col in wrist_cols:
            if col.endswith("_pos"):
                values = df[col][df[col] != -9999]
                if len(values) > 0:
                    print(f"  {col}: min={values.min()}, max={values.max()}, std={values.std():.2f}")

        # Check for high loads
        load_col = "follower_wrist_flex_load"
        if load_col in df.columns:
            high_loads = df[df[load_col] > 800][load_col]
            if len(high_loads) > 0:
                print(f"  ⚠️  High load events: {len(high_loads)} samples above 800")

        # Check for large position differences
        diff_col = "diff_wrist_flex_pos"
        if diff_col in df.columns:
            large_diffs = df[abs(df[diff_col]) > 500][diff_col]
            if len(large_diffs) > 0:
                print(f"  ⚠️  Large position differences: {len(large_diffs)} samples > 500")

    except ImportError:
        print("📊 Install pandas for detailed analysis: pip install pandas")
    except Exception as e:
        print(f"❌ Analysis error: {e}")


if __name__ == "__main__":
    # Your port configurations
    LEADER_PORT = "/dev/tty.usbmodem58CD1771421"
    FOLLOWER_PORT = "/dev/tty.usbmodem58760434091"

    logger = JointLogger(LEADER_PORT, FOLLOWER_PORT)

    try:
        logger.run()

        # Quick analysis if pandas is available
        if logger.csv_file.exists():
            analyze_csv(logger.csv_file)

    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        logger.disconnect()
