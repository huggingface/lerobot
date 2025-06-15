import time
import os
from dataclasses import asdict, dataclass
from pynput import keyboard
from lerobot.common.robots import (  # noqa: F401
    make_robot_from_config,
    RobotConfig,
    Robot,
)
import draccus


@dataclass
class TeleoperateConfig:
    robot: RobotConfig


def generate_chess_positions():
    """Generate all chess board positions from A1 to H8."""
    positions = []
    for col in "ABCDEFGH":
        for row in range(1, 9):
            positions.append(f"{col}{row}")
    return positions


def clear_screen():
    """Clear the terminal screen."""
    os.system("cls" if os.name == "nt" else "clear")


class PositionRecorder:
    def __init__(self, robot: Robot):
        self.positions = generate_chess_positions()
        self.current_index = 0
        self.running = True
        self.listener = None
        self.robot = robot

    def on_press(self, key):
        try:
            if (
                key == keyboard.Key.space
                or key == keyboard.Key.enter
                or key == keyboard.Key.right
            ):
                print(f"\nRecording position {self.positions[self.current_index]}...")
                observation = self.robot.get_observation()
                print(f"Observation: {observation}")
                time.sleep(0.3)  # Prevent multiple triggers
                if self.current_index < len(self.positions) - 1:
                    self.current_index += 1
            elif key == keyboard.Key.left:
                if self.current_index > 0:
                    self.current_index -= 1
                    print(f"\nGoing back to {self.positions[self.current_index]}...")
                    time.sleep(0.3)
            elif key == keyboard.Key.esc:
                print("\nExiting...")
                self.running = False
                return False  # Stop listener
        except AttributeError:
            pass  # Ignore other keys

    def display_status(self):
        clear_screen()
        current_pos = self.positions[self.current_index]

        print(f"\nCurrent Position: {current_pos}")
        print(f"Progress: {self.current_index + 1}/{len(self.positions)}")
        print("\nControls:")
        print("- SPACE: Record and continue")
        print("- LEFT ARROW: Go back")
        print("- ESC: Exit")

    def run(self):
        print("Chess Board Position Recorder")
        print("Controls:")
        print("- SPACE: Record current position and move to next")
        print("- LEFT ARROW: Go back to previous position")
        print("- ESC: Exit")
        print("\nPress Enter to start...")

        # Wait for Enter key to start
        input()

        # Start the main listener
        with keyboard.Listener(on_press=self.on_press) as listener:
            self.listener = listener
            while self.running:
                self.display_status()
                time.sleep(0.1)  # Small delay to prevent high CPU usage


@draccus.wrap()
def record_board(cfg: TeleoperateConfig):
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    recorder = PositionRecorder(robot)
    recorder.run()


if __name__ == "__main__":
    record_board()
