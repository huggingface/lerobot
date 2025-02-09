import os
import time
from contextlib import suppress
from dataclasses import dataclass
from functools import cache
from tkinter import Button, Canvas, Frame, Label, Tk
from typing import List, Optional


@cache
def check_gui_env():
    return os.getenv("LEROBOT_GUI", "False").lower() in ["true", "1", "t"]


@dataclass
class UserAction:
    message: str
    message_size: int
    button_text: str
    next_action: Optional["UserAction"] = None
    is_final: Optional[bool] = False


class PromptGUI:
    def __init__(self, title: str = None):
        self.root = Tk()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        if title:
            self.root.title(title)
        self.label = Label(self.root, text="", justify="left", font=("Helvetica", 24), wraplength=400)
        self.label.pack(fill="both", expand=True)
        self.button = None
        self.next_step = lambda: ()
        self.steps: List[UserAction] = []
        self.done = False

    def show_message(self, message, font=("Helvetica", 24)):
        self.label.config(text=message, font=font)

    def ask_for_input(self, prompt: str, size: int = 24, button_text: str = "OK"):
        """Ask for user input with a prompt and button text."""
        self.show_message(prompt, font=("Helvetica", size))
        if not self.button:
            self.button = Button(self.root, text=button_text, command=self.next_step, font=("Helvetica", 36))
            self.button.pack(fill="x")
        else:
            self.button.config(command=self.next_step, text=button_text)

    def cleanup(self):
        self.root.quit()

    def add_step(self, prompt: str, prompt_size=24, button_text="OK", is_final=False):
        step = UserAction(prompt, prompt_size, button_text, is_final=is_final)
        if len(self.steps) > 0:
            self.steps[-1].next_action = step
        self.steps.append(step)

    def display_next_step(self):
        if len(self.steps) == 1:
            self.next_step = self.cleanup
        else:
            self.next_step = self.display_next_step
        step = self.steps[0]
        if step.is_final:
            self.next_step = self.terminate
        self.ask_for_input(step.message, step.message_size, step.button_text)
        self.steps.pop(0)

    def run(self):
        self.display_next_step()
        self.root.mainloop()

    def terminate(self):
        self.done = True
        self.root.destroy()


class RecordingGUI:
    def __init__(self, title: str, events: dict):
        self.root = Tk()
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}")
        if title:
            self.root.title(title)
        self.label = Label(self.root, text="", justify="left", font=("Helvetica", 24), wraplength=400)
        self.label.pack(fill="both", expand=True)
        self.button_frame = Frame(self.root)
        self.button_frame.pack(fill="x")
        self.next_button = Button(
            self.button_frame, text="Next", command=self.on_next_click, font=("Helvetica", 36)
        )
        self.reset_button = Button(
            self.button_frame, text="Reset", command=self.on_reset_click, font=("Helvetica", 36)
        )
        self.stop_button = Button(
            self.button_frame, text="Stop", command=self.on_stop_click, font=("Helvetica", 36)
        )
        self.next_button.pack(side="left", fill="x", expand=True)
        self.reset_button.pack(side="left", fill="x", expand=True)
        self.stop_button.pack(side="left", fill="x", expand=True)
        self.progress_frame = Frame(self.root)
        self.progress_frame.pack(fill="x")
        self.canvas = Canvas(self.progress_frame, width=400, height=40)
        self.canvas.pack()
        self.last_clicked = None
        self.remaining_time = 0
        self.start_time = 0
        self._events = events
        self.root.bind("<Right>", lambda event: self.on_next_click())
        self.root.bind("<Left>", lambda event: self.on_reset_click())
        self.root.bind("<Escape>", lambda event: self.on_stop_click())

    def show_message(self, message: str):
        """Display a message on the GUI."""
        self.label.config(text=message)

    def reset_progress_bar(self):
        self.canvas.delete("all")
        self.canvas.create_rectangle(0, 0, 400, 20, fill="red")

    def update_progress_bar(self):
        """Update the progress bar based on the remaining time."""
        if self.start_time != 0:
            elapsed_time = time.time() - self.start_time
            if elapsed_time < self.remaining_time:
                progress_width = int(400 * (1 - (elapsed_time / self.remaining_time)))
                self.canvas.delete("all")
                self.canvas.create_rectangle(0, 0, progress_width, 20, fill="blue")
                self.root.after(100, self.update_progress_bar)
            else:
                self.reset_progress_bar()

    def cleanup(self):
        self.reset_progress_bar()
        self.root.update()
        self.root.quit()
        self.update_events()

    def set_remaining_time(self, seconds: int):
        """Set the remaining time and start updating the progress bar."""
        self.remaining_time = seconds
        self.start_time = time.time()
        self.update_progress_bar()

    def on_next_click(self):
        """Handle next button click or right arrow key press."""
        self.last_clicked = "Next"
        self.show_message("Next...")
        self.cleanup()

    def on_reset_click(self):
        """Handle reset button click or left arrow key press."""
        self.last_clicked = "Reset"
        self.show_message("Resetting...")
        self.cleanup()

    def on_stop_click(self):
        """Handle stop button click or Esc key press."""
        self.last_clicked = "Stop"
        self.show_message("Exiting")
        self.cleanup()

    def update_events(self):
        self._events["exit_early"] = self.last_clicked == "Next"
        self._events["rerecord_episode"] = self.last_clicked == "Reset"
        self._events["stop_recording"] = self.last_clicked == "Stop"
        return

    def terminate(self):
        with suppress(Exception):  # in case GUI is already terminating
            self.root.destroy()

    def run(self, remaining_time: int):
        """Run the GUI application and return the last clicked button."""
        self.set_remaining_time(remaining_time)
        self.root.mainloop()
        with suppress(Exception):  # in case GUI is already terminating
            self.cleanup()
        self.last_clicked = None
        return
