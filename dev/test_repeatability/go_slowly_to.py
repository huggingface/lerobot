#!/usr/bin/env python3
"""
Follower Position Playback Script
Reads recorded positions from JSON and makes follower go through them slowly with verification.

Timeline for each position:
- 3 seconds: Interpolated movement to target position
- 1 second: Stay at position (stabilization)
- Position verification: Compare actual vs expected position
- 1 more second: Continue staying at position
"""

import json
import time
import sys
import signal
import glob
from typing import Dict, List, Any
from lerobot.common.robots.so101_follower import SO101FollowerConfig, SO101Follower

# Configuration
FOLLOWER_PORT = "/dev/ttyACM0"
MOVEMENT_TIME = 3.0  # seconds to move between positions
STABILIZATION_TIME = 3.0  # seconds to wait before verification
HOLD_TIME = 1.0  # seconds to hold after verification
INTERPOLATION_STEPS = 3000  # number of steps for smooth movement

class PositionPlayback:
    def __init__(self):
        self.robot_config = SO101FollowerConfig(
            port=FOLLOWER_PORT,
            id="follower10",
        )
        self.robot = SO101Follower(self.robot_config)
        self.recordings = []
        self.current_position = {}
        self.connected = False
        
    def cleanup_and_exit(self, signum=None, frame=None):
        """Clean shutdown."""
        print("\nShutting down...")
        try:
            if self.connected:
                self.robot.bus.disable_torque()
                self.robot.disconnect()
                print("✓ Robot disconnected")
            else:
                print("Robot was not connected")
        except Exception as e:
            print(f"Cleanup error: {e}")
        sys.exit(0)
    
    def load_recordings(self, filename: str) -> bool:
        """Load recordings from JSON file."""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            self.recordings = data['recordings']
            print(f"✓ Loaded {len(self.recordings)} recordings from {filename}")
            
            # Display recording info
            print(f"Recording timestamp: {data.get('timestamp', 'Unknown')}")
            print(f"Total recordings: {data.get('total_recordings', len(self.recordings))}")
            
            return True
            
        except Exception as e:
            print(f"✗ Failed to load recordings: {e}")
            return False
    
    def find_latest_recording_file(self) -> str:
        """Find the most recent recording file."""
        # Check both dev and test_repeatability directories
        patterns = [
            "/home/tianchongj/workspace/lerobot/dev/teleop_recordings_*.json",
            "/home/tianchongj/workspace/lerobot/dev/test_repeatability/teleop_recordings_*.json"
        ]
        
        all_files = []
        for pattern in patterns:
            files = glob.glob(pattern)
            all_files.extend(files)
        
        if not all_files:
            raise FileNotFoundError("No recording files found. Please run teleop_record.py first.")
        
        # Sort by filename (which includes timestamp)
        latest_file = sorted(all_files)[-1]
        print(f"Using latest recording file: {latest_file}")
        return latest_file
    
    def interpolate_positions(self, start_pos: Dict[str, int], end_pos: Dict[str, int], steps: int) -> List[Dict[str, int]]:
        """Generate interpolated positions between start and end."""
        interpolated = []
        
        for i in range(steps + 1):
            alpha = i / steps  # 0 to 1
            interp_pos = {}
            
            for motor_name in start_pos:
                if motor_name in end_pos:
                    start_val = start_pos[motor_name]
                    end_val = end_pos[motor_name]
                    interp_val = int(start_val + alpha * (end_val - start_val))
                    interp_pos[motor_name] = interp_val
            
            interpolated.append(interp_pos)
        
        return interpolated
    
    def move_to_position_smoothly(self, target_position: Dict[str, int], duration: float):
        """Move to target position with smooth interpolation."""
        print(f"Moving to position over {duration:.1f} seconds...")
        
        # Get current position
        try:
            current_pos = self.robot.bus.sync_read("Present_Position", normalize=False)
        except Exception as e:
            print(f"Failed to read current position: {e}")
            return False
        
        # Generate interpolated path
        interpolated_path = self.interpolate_positions(current_pos, target_position, INTERPOLATION_STEPS)
        
        # Execute smooth movement
        step_duration = duration / INTERPOLATION_STEPS
        
        for i, step_position in enumerate(interpolated_path):
            try:
                self.robot.bus.sync_write("Goal_Position", step_position, normalize=False)
                
                # Progress indicator
                progress = (i + 1) / len(interpolated_path) * 100
                print(f"\rProgress: {progress:5.1f}%", end="", flush=True)
                
                time.sleep(step_duration)
                
            except Exception as e:
                print(f"\nMovement error at step {i}: {e}")
                return False
        
        print()  # New line after progress
        return True
    
    def verify_position(self, expected_follower: Dict[str, int], expected_leader: Dict[str, int], recording_id: int):
        """Print position differences for both follower and leader."""
        try:
            actual_follower = self.robot.bus.sync_read("Present_Position", normalize=False)
            
            print(f"\n--- Position Differences (Recording #{recording_id}) ---")
            
            # Compare follower positions
            print("FOLLOWER COMPARISON:")
            for motor_name in expected_follower:
                if motor_name in actual_follower:
                    expected = expected_follower[motor_name]
                    actual = actual_follower[motor_name]
                    difference = actual - expected
                    print(f"  {motor_name:12}: Expected {expected:4d}, Actual {actual:4d}, Diff {difference:+4d}")
            
            # Show leader positions from recording for reference
            print("LEADER REFERENCE (from recording):")
            for motor_name in expected_leader:
                leader_pos = expected_leader[motor_name]
                print(f"  {motor_name:12}: {leader_pos:4d}")
            
            # Compare follower actual vs leader recorded
            print("FOLLOWER vs LEADER DIFFERENCE:")
            for motor_name in expected_leader:
                if motor_name in actual_follower:
                    leader_pos = expected_leader[motor_name]
                    follower_pos = actual_follower[motor_name]
                    difference = follower_pos - leader_pos
                    print(f"  {motor_name:12}: Leader {leader_pos:4d}, Follower {follower_pos:4d}, Diff {difference:+4d}")
            
        except Exception as e:
            print(f"Position verification failed: {e}")
    
    def run_playback(self):
        """Execute the complete playback sequence."""
        if not self.recordings:
            print("No recordings loaded")
            return
        
        print(f"\n{'='*60}")
        print("STARTING POSITION PLAYBACK")
        print(f"{'='*60}")
        print(f"Total positions to visit: {len(self.recordings)}")
        print(f"Movement time per position: {MOVEMENT_TIME}s")
        print(f"Hold time per position: {STABILIZATION_TIME + HOLD_TIME}s")
        print(f"Total estimated time: {len(self.recordings) * (MOVEMENT_TIME + STABILIZATION_TIME + HOLD_TIME):.1f}s")
        print(f"{'='*60}")
        
        for i, recording in enumerate(self.recordings):
            recording_id = recording.get('recording_id', i + 1)
            follower_target = recording['follower_positions']
            leader_reference = recording['leader_positions']
            
            print(f"\n--- Recording #{recording_id} ({i+1}/{len(self.recordings)}) ---")
            print(f"Follower Target: {follower_target}")
            print(f"Leader Reference: {leader_reference}")
            
            # Step 1: Move to position (3 seconds with interpolation)
            success = self.move_to_position_smoothly(follower_target, MOVEMENT_TIME)
            if not success:
                print("Movement failed, continuing to next position...")
                continue
            
            # Step 2: Wait for stabilization (3 seconds)
            print(f"Stabilizing for {STABILIZATION_TIME}s...")
            time.sleep(STABILIZATION_TIME)
            
            # Step 3: Print position differences
            self.verify_position(follower_target, leader_reference, recording_id)
            
            # Step 4: Hold position (1 more second)
            print(f"Holding position for {HOLD_TIME}s...")
            time.sleep(HOLD_TIME)
        
        # Final summary
        print(f"\n{'='*60}")
        print("PLAYBACK COMPLETE")
        print(f"{'='*60}")
        print(f"Total positions: {len(self.recordings)}")
        print(f"{'='*60}")

def main():
    playback = PositionPlayback()
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, playback.cleanup_and_exit)
    signal.signal(signal.SIGTERM, playback.cleanup_and_exit)
    
    try:
        # Find and load recording file
        if len(sys.argv) > 1:
            recording_file = sys.argv[1]
        else:
            recording_file = playback.find_latest_recording_file()
        
        if not playback.load_recordings(recording_file):
            return 1
        
        # Connect to robot
        print("Connecting to follower robot...")
        playback.robot.connect(calibrate=True)
        playback.connected = True
        print("✓ Connected to follower")
        
        # Run playback
        playback.run_playback()
        
    except KeyboardInterrupt:
        print("\nPlayback interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        playback.cleanup_and_exit()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
