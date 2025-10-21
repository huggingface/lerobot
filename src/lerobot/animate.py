#!/usr/bin/env python3
"""
SO-ARM101 Simple Animation Tool
==============================

A simplified script for manual animation with the SO-ARM101 robotic arm.

Usage:
1. Run the script
2. Manually move your robot arm to desired positions
3. Press ENTER to record each pose
4. Type 'play' to play back your animation
5. Type 'save' to save your animation

Requirements:
- pip install lerobot
- SO-ARM101 follower arm connected and calibrated
"""

import json
import time
import sys
from pathlib import Path
try:
    from lerobot.robots import make_robot_from_config
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    LEROBOT_AVAILABLE = True
except ImportError:
    LEROBOT_AVAILABLE = False
    print("ERROR: LeRobot not installed correctly!")
    print("Install with: pip install lerobot")
    sys.exit(1)


class SimpleAnimator:
    def __init__(self, port):
        self.port = port
        self.robot = None
        self.poses = []  # List of recorded poses
        self.speed = 1.0  # Default speed multiplier
        self.delay = 0.5  # Default delay between poses
        # Motor names for SO101 - matching the actual motor configuration
        self.motor_names = [
            'shoulder_pan', 'shoulder_lift', 'elbow_flex', 
            'wrist_flex', 'wrist_roll', 'gripper'
        ]
        
        # Create animations directory
        self.anim_dir = Path("so101_animations")
        self.anim_dir.mkdir(exist_ok=True)
        
        print(f"SO-ARM101 Simple Animator")
        print(f"Animation files will be saved to: {self.anim_dir}")
    
    def connect(self):
        """Connect to the robot"""
        if not LEROBOT_AVAILABLE:
            print("ERROR: LeRobot not installed!")
            print("Install with: pip install lerobot")
            return False
        
        try:
            print(f"Connecting to robot on {self.port}...")
            # Create robot configuration
            config = SO101FollowerConfig(
                port=self.port,
                use_degrees=True
            )
            # Create robot instance
            self.robot = make_robot_from_config(config)
            self.robot.connect()
            print("✓ Connected successfully!")
            
            # Disable torque so the arm can be moved manually
            print("Disabling torque for manual control...")
            self.robot.bus.disable_torque()
            print("✓ Torque disabled - you can now move the arm manually")
            
            return True
        except Exception as e:
            print(f"✗ Connection failed: {e}")
            print("Make sure your robot is:")
            print("- Connected via USB")
            print("- Properly calibrated")
            print("- Using the correct port")
            return False
    
    def get_current_pose(self):
        """Get current motor positions"""
        try:
            # Get current observation using the correct method
            obs = self.robot.get_observation()
            # Extract motor positions
            pose = {}
            for motor in self.motor_names:
                key = f"{motor}.pos"
                if key in obs:
                    pose[motor] = float(obs[key])
                else:
                    pose[motor] = 0.0
            return pose
        except Exception as e:
            print(f"Error reading pose: {e}")
            return None
    
    def set_pose(self, pose, duration=2.0):
        """Move robot to a specific pose"""
        try:
            # First enable torque to control the motors
            self.robot.bus.enable_torque()
            
            # Convert pose to action dictionary (using .pos suffix as expected by send_action)
            action = {}
            for motor in self.motor_names:
                if motor in pose:
                    action[f"{motor}.pos"] = pose[motor]
            
            # Send action to robot
            self.robot.send_action(action)
            time.sleep(duration)
                
        except Exception as e:
            print(f"Error setting pose: {e}")
    
    def record_pose(self):
        """Record current robot pose"""
        pose = self.get_current_pose()
        if pose is not None:
            self.poses.append(pose)
            print(f"✓ Recorded pose #{len(self.poses)}")
            self.print_pose(pose)
            # Disable torque so the arm can be moved manually again
            try:
                self.robot.bus.disable_torque()
            except:
                pass
            return True
        return False
    
    def print_pose(self, pose):
        """Print pose in a readable format"""
        print("  Motor positions:")
        for motor, value in pose.items():
            print(f"    {motor:12}: {value:7.2f}")
    
    def play_animation(self, loop=False, custom_speed=None, custom_delay=None):
        """Play recorded animation using current or custom speed and delay settings"""
        if not self.poses:
            print("No poses recorded!")
            return
        
        # Use custom values if provided, otherwise use instance settings
        speed = custom_speed if custom_speed is not None else self.speed
        delay = custom_delay if custom_delay is not None else self.delay
        
        print(f"\n🎬 Playing animation ({len(self.poses)} poses, speed={speed}x, delay={delay}s)")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                for i, pose in enumerate(self.poses):
                    print(f"Moving to pose {i+1}/{len(self.poses)}")
                    
                    # Calculate transition time based on speed (much shorter base time)
                    transition_time = 0.5 / speed  # Base 0.5 seconds instead of 2.0
                    if i == 0:
                        transition_time = 0.3 / speed  # Even faster for first pose
                    
                    self.set_pose(pose, transition_time)
                    
                    # Delay between poses (except for last pose)
                    if i < len(self.poses) - 1:
                        time.sleep(delay)
                
                if not loop:
                    break
                    
                print("Looping...")
                time.sleep(1.0)  # Fixed 1 second delay before loop restart
                
        except KeyboardInterrupt:
            print("\n⏹️ Animation stopped")
        finally:
            # Disable torque after playback so arm can be moved manually
            try:
                self.robot.bus.disable_torque()
            except:
                pass
    
    def save_animation(self, name=None):
        """Save animation to file"""
        if not self.poses:
            print("No poses to save!")
            return
        
        if name is None:
            name = input("Enter animation name (or press Enter for auto): ").strip()
            if not name:
                name = f"animation_{int(time.time())}"
        
        filename = f"{name}.json"
        filepath = self.anim_dir / filename
        
        data = {
            'name': name,
            'created': time.strftime('%Y-%m-%d %H:%M:%S'),
            'poses': self.poses,
            'motor_names': self.motor_names
        }
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✓ Animation saved: {filepath}")
        except Exception as e:
            print(f"✗ Save failed: {e}")
    
    def load_animation(self, filename):
        """Load animation from file"""
        filepath = self.anim_dir / filename
        if not filepath.exists():
            filepath = Path(filename)  # Try direct path
        
        if not filepath.exists():
            print(f"File not found: {filename}")
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.poses = data['poses']
            print(f"✓ Loaded animation: {data.get('name', filename)}")
            print(f"  Poses: {len(self.poses)}")
            print(f"  Created: {data.get('created', 'Unknown')}")
            return True
        except Exception as e:
            print(f"✗ Load failed: {e}")
            return False
    
    def list_animations(self):
        """List saved animations and return list of files"""
        animations = list(self.anim_dir.glob("*.json"))
        if not animations:
            print("No saved animations found.")
            return []
        
        print(f"\n📁 Saved animations in {self.anim_dir}:")
        for i, anim_file in enumerate(animations, 1):
            try:
                with open(anim_file, 'r') as f:
                    data = json.load(f)
                name = data.get('name', anim_file.stem)
                poses = len(data.get('poses', []))
                created = data.get('created', 'Unknown')
                print(f"  {i}. {name}")
                print(f"     File: {anim_file.name}, Poses: {poses}, Created: {created}")
            except:
                print(f"  {i}. {anim_file.name} (corrupted)")
        
        return animations
    
    def select_and_load_animation(self):
        """Interactive menu to select and load an animation"""
        animations = self.list_animations()
        if not animations:
            return False
        
        try:
            choice = input("\nEnter number to load (or 'c' to cancel): ").strip()
            if choice.lower() == 'c':
                return False
            
            idx = int(choice) - 1
            if 0 <= idx < len(animations):
                return self.load_animation(animations[idx].name)
            else:
                print("Invalid selection")
                return False
        except ValueError:
            print("Invalid input")
            return False
    
    def play_saved_animation(self):
        """Load and play a saved animation with menu"""
        if self.select_and_load_animation():
            print("\nAnimation loaded! Choose playback option:")
            print("1. Play normal speed")
            print("2. Play with custom speed")
            print("3. Play in loop")
            print("4. Cancel")
            
            choice = input("Enter choice (1-4): ").strip()
            
            if choice == '1':
                self.play_animation()
            elif choice == '2':
                try:
                    speed = float(input("Enter speed (0.1-5.0): "))
                    delay = float(input("Enter delay between poses (0-3.0): "))
                    self.play_animation(speed=speed, delay=delay)
                except ValueError:
                    print("Invalid input")
            elif choice == '3':
                self.play_animation(loop=True)
            elif choice == '4':
                print("Playback cancelled")
    
    def clear_poses(self):
        """Clear all recorded poses"""
        self.poses.clear()
        print("✓ All poses cleared")
    
    def show_help(self):
        """Show help information"""
        print("\n" + "="*50)
        print("SO-ARM101 Simple Animator - Commands")
        print("="*50)
        print("Recording:")
        print("  ENTER     - Record current pose")
        print("  clear     - Clear all recorded poses")
        print("  status    - Show recorded poses count")
        print()
        print("Playback (Current Animation):")
        print("  play      - Play current animation")
        print("  loop      - Play current animation in loop")
        print()
        print("Settings:")
        print("  speed X   - Set animation speed (e.g., speed 0.5, speed 2)")
        print("  delay X   - Set delay between poses (e.g., delay 1, delay 0.2)")
        print()
        print("File Operations:")
        print("  save      - Save current animation")
        print("  load      - Load animation from file (interactive menu)")
        print("  library   - Play from saved animations library")
        print("  list      - List saved animations")
        print()
        print("Other:")
        print("  help      - Show this help")
        print("  quit/exit - Exit program")
        print("="*50)
    
    def run(self):
        """Main interaction loop"""
        print("\n🤖 SO-ARM101 Simple Animator Ready!")
        print("Type 'help' for commands, 'quit' to exit")
        print("\nTo record poses:")
        print("1. Manually move your robot arm to desired position")
        print("2. Press ENTER to record the pose")
        print("3. Repeat for multiple poses")
        print("4. Type 'play' to see your animation")
        
        while True:
            try:
                # Show current status
                status = f"[{len(self.poses)} poses | speed={self.speed}x | delay={self.delay}s]"
                command = input(f"\n{status} > ").strip()
                
                # Split command for multi-word commands
                parts = command.split()
                cmd = parts[0].lower() if parts else ''
                
                if command == '' or cmd == 'record':
                    self.record_pose()
                
                elif cmd == 'play':
                    self.play_animation()
                
                elif cmd == 'loop':
                    self.play_animation(loop=True)
                
                elif cmd == 'speed':
                    if len(parts) > 1:
                        try:
                            new_speed = float(parts[1])
                            if 0.1 <= new_speed <= 5.0:
                                self.speed = new_speed
                                print(f"✓ Speed set to {self.speed}x")
                            else:
                                print("Speed must be between 0.1 and 5.0")
                        except ValueError:
                            print("Invalid speed value")
                    else:
                        print(f"Current speed: {self.speed}x")
                        print("Usage: speed <value> (e.g., speed 0.5, speed 2)")
                
                elif cmd == 'delay':
                    if len(parts) > 1:
                        try:
                            new_delay = float(parts[1])
                            if 0 <= new_delay <= 5.0:
                                self.delay = new_delay
                                print(f"✓ Delay set to {self.delay}s")
                            else:
                                print("Delay must be between 0 and 5.0 seconds")
                        except ValueError:
                            print("Invalid delay value")
                    else:
                        print(f"Current delay: {self.delay}s")
                        print("Usage: delay <value> (e.g., delay 1, delay 0.2)")
                
                elif cmd == 'save':
                    self.save_animation()
                
                elif cmd == 'load':
                    self.select_and_load_animation()
                
                elif cmd == 'library':
                    self.play_saved_animation()
                
                elif cmd == 'list':
                    self.list_animations()
                
                elif cmd == 'clear':
                    confirm = input("Clear all poses? (y/n): ").lower()
                    if confirm == 'y':
                        self.clear_poses()
                
                elif cmd == 'status':
                    print(f"Recorded poses: {len(self.poses)}")
                    print(f"Speed: {self.speed}x")
                    print(f"Delay: {self.delay}s")
                    if self.poses:
                        print("Latest pose:")
                        self.print_pose(self.poses[-1])
                
                elif cmd == 'help':
                    self.show_help()
                
                elif cmd in ['quit', 'exit', 'q']:
                    break
                
                else:
                    print(f"Unknown command: {command}")
                    print("Type 'help' for available commands")
            
            except KeyboardInterrupt:
                print("\nExiting...")
                break
            except Exception as e:
                print(f"Error: {e}")
        
        # Cleanup
        if self.robot:
            try:
                self.robot.disconnect()
            except:
                pass
        
        print("👋 Goodbye!")


def main():
    """Main function with CLI argument parsing"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="SO-ARM101 Animation Tool - Create and play animations with your robot arm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  lerobot-animate /dev/tty.usbmodem5A680123451
  lerobot-animate /dev/ttyUSB0
  lerobot-animate COM3
  
To find your port, run:
  lerobot-find-port
        """
    )
    
    parser.add_argument(
        "port",
        type=str,
        help="Serial port for the robot arm (e.g., /dev/tty.usbmodem5A680123451)"
    )
    
    parser.add_argument(
        "--load",
        type=str,
        metavar="FILE",
        help="Load and play an animation file on startup"
    )
    
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Loop animation playback"
    )
    
    args = parser.parse_args()
    
    print("SO-ARM101 Animation Tool")
    print("========================")
    print(f"Using port: {args.port}")
    
    # Create animator with specified port
    animator = SimpleAnimator(args.port)
    
    if animator.connect():
        # Load animation if specified
        if args.load:
            if animator.load_animation(args.load):
                print(f"Auto-loaded animation: {args.load}")
                if input("Play now? (y/n): ").lower() == 'y':
                    animator.play_animation(loop=args.loop)
        
        # Run interactive mode
        animator.run()
    else:
        print("\nFailed to connect to robot. Please check:")
        print("1. Robot is connected via USB")
        print("2. Correct port is specified")
        print("3. Robot is calibrated with LeRobot")
        print("4. LeRobot is properly installed")
        print("\nTo find your port, run: lerobot-find-port")


if __name__ == "__main__":
    main()