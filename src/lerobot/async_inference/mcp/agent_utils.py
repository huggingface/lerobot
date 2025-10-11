#!/usr/bin/env python3
"""
Utility functions for the AI Agent.
Contains image display and other helper functionality.
"""

import base64
import io
import math
import multiprocessing
import time
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ImageGridViewer:
    """GUI window that displays images in a dynamic grid."""
    
    def __init__(self, image_queue):
        self.image_queue = image_queue
        self.images = []
        self.labels = []
        self.current_grid_size = 0
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Robot Camera Feed")
        self.root.geometry("800x600")
        self.root.attributes('-topmost', True)
        self.root.after(1000, lambda: self.root.attributes('-topmost', False))
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add status label
        self.status_label = ttk.Label(self.main_frame, text="Waiting for images...")
        self.status_label.pack(expand=True)
        
        # Start checking for images
        self.check_queue()
    
    def calculate_grid_size(self, num_images):
        """Calculate NxN grid where N^2 >= num_images."""
        if num_images == 0:
            return 0
        return math.ceil(math.sqrt(num_images))
    
    def update_grid(self):
        """Update the grid layout with current images."""
        num_images = len(self.images)
        
        # Hide status label when we have images
        if num_images > 0 and self.status_label:
            self.status_label.destroy()
            self.status_label = None
        
        if num_images == 0:
            return
        
        grid_size = self.calculate_grid_size(num_images)
        
        # Rebuild grid if size changed
        if grid_size != self.current_grid_size:
            # Clear existing labels
            for label in self.labels:
                label.destroy()
            self.labels = []
            
            # Create new grid
            for i in range(grid_size * grid_size):
                label = ttk.Label(self.main_frame)
                row = i // grid_size
                col = i % grid_size
                label.grid(row=row, column=col, padx=5, pady=5, sticky="nsew")
                self.labels.append(label)
            
            # Configure grid weights
            for i in range(grid_size):
                self.main_frame.grid_rowconfigure(i, weight=1)
                self.main_frame.grid_columnconfigure(i, weight=1)
            
            self.current_grid_size = grid_size
        
        # Update images
        cell_width = max(100, 800 // grid_size - 20)
        cell_height = max(100, 600 // grid_size - 20)
        
        for i, image_data in enumerate(self.images):
            if i < len(self.labels):
                try:
                    img_bytes = base64.b64decode(image_data)
                    pil_img = Image.open(io.BytesIO(img_bytes))
                    pil_img.thumbnail((cell_width, cell_height), Image.Resampling.LANCZOS)
                    tk_img = ImageTk.PhotoImage(pil_img)
                    
                    self.labels[i].configure(image=tk_img)
                    self.labels[i].image = tk_img  # Keep reference
                except Exception as e:
                    print(f"Error displaying image {i}: {e}")
        
        # Clear unused labels
        for i in range(len(self.images), len(self.labels)):
            self.labels[i].configure(image="")
            self.labels[i].image = None
    
    def check_queue(self):
        """Check for new images from main process."""
        try:
            while True:
                try:
                    images_data = self.image_queue.get_nowait()
                    if images_data == "QUIT":
                        self.root.quit()
                        return
                    self.images = images_data
                    self.update_grid()
                except:
                    break
        except:
            pass
        
        # Schedule next check
        self.root.after(100, self.check_queue)
    
    def run(self):
        """Run the main GUI loop."""
        self.root.mainloop()


def image_grid_viewer_process(image_queue):
    """Process function that runs the image viewer."""

    try:
        viewer = ImageGridViewer(image_queue)
        viewer.run()
    except Exception as e:
        print(f"📸 Image viewer error: {e}")


class ImageViewer:
    """Manages the image display window in a separate process."""
    
    def __init__(self):
        self.image_queue = multiprocessing.Queue()
        self.image_viewer_process = None
        self.current_images = []
    
    def start(self):
        """Start the image viewer process."""
        if self.image_viewer_process is None or not self.image_viewer_process.is_alive():
            self.image_viewer_process = multiprocessing.Process(
                target=image_grid_viewer_process,
                args=(self.image_queue,),
                daemon=True
            )
            self.image_viewer_process.start()
            print("📸 Image viewer window opened")
            time.sleep(0.5)
    
    def update(self, image_parts):
        """Update the viewer with new images."""
        if not image_parts:
            return
        
        self.start()
        
        # Extract image data
        new_images = []
        for image_part in image_parts:
            if image_part.get('source', {}).get('data'):
                new_images.append(image_part['source']['data'])
        
        if new_images:
            self.current_images = new_images
            try:
                self.image_queue.put(self.current_images)
                print(f"📸 Updated image viewer with {len(new_images)} images")
            except Exception as e:
                print(f"📸 Error updating image viewer: {e}")
    
    def cleanup(self):
        """Clean up the image viewer process."""
        if self.image_viewer_process and self.image_viewer_process.is_alive():
            print("📸 Closing image viewer...")
            try:
                self.image_queue.put("QUIT")
                self.image_viewer_process.join(timeout=2)
                if self.image_viewer_process.is_alive():
                    self.image_viewer_process.terminate()
                    self.image_viewer_process.join(timeout=1)
                print("📸 Image viewer closed")
            except Exception as e:
                print(f"📸 Error closing image viewer: {e}") 