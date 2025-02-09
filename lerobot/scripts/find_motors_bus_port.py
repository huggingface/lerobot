import os
import time
from pathlib import Path

from serial.tools import list_ports  # Part of pyserial library

from lerobot.common.utils.gui import PromptGUI


def find_available_ports():
    if os.name == "nt":  # Windows
        # List COM ports using pyserial
        ports = [port.device for port in list_ports.comports()]
    else:  # Linux/macOS
        # List /dev/tty* ports for Unix-based systems
        ports = [str(path) for path in Path("/dev").glob("tty*")]
    return ports


def find_port_core(display_message, wait_for_input):
    display_message("Finding all available ports for the MotorsBus.")
    ports_before = find_available_ports()
    display_message(f"Ports before disconnecting: {ports_before}", 14)
    display_message("Remove the USB cable from your MotorsBus and press 'OK' when done.")
    wait_for_input()
    time.sleep(0.5)  # Allow some time for port to be released
    ports_after = find_available_ports()
    ports_diff = list(set(ports_before) - set(ports_after))
    if len(ports_diff) == 1:
        port = ports_diff[0]
        display_message(f"The port of this MotorsBus is '{port}'")
        display_message("Reconnect the USB cable.", button="Exit")
        wait_for_input()
        return port
    elif len(ports_diff) == 0:
        raise OSError(f"Could not detect the port. No difference was found ({ports_diff}).")
    else:
        raise OSError(f"Could not detect the port. More than one port was found ({ports_diff}).")


def find_port_with_gui():
    gui = PromptGUI("Motors Bus Port Detector")

    def display_message(message, size=24, button="OK"):
        gui.add_step(message, size, button)

    def wait_for_input():
        gui.run()

    try:
        find_port_core(display_message, wait_for_input)
    except OSError as e:
        display_message(e)
        wait_for_input()
    finally:
        gui.terminate()


def find_port_with_console():
    return find_port_core(print, input)


def find_port():
    if os.getenv("LEROBOT_GUI", "False").lower() in ["true", "1", "t"]:
        return find_port_with_gui()
    else:
        return find_port_with_console()


if __name__ == "__main__":
    # Helper to find the USB port associated with your MotorsBus.
    find_port()
