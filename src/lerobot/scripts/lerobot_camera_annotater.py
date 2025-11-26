#!/usr/bin/env python3
"""Interactive camera annotation tool for drawing numbered squares on live video feeds."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2

from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig


class CameraState:
    """Annotation state for a single camera."""
    
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.boxes = []  # list of [x0, y0, x1, y1, id] (mutable for moving)
        self.next_id = 1
        self.drawing = False
        self.start_pt = None
        self.current_box = None
        self.recording = False
        self.writer = None
        # Moving state
        self.moving_idx = None  # index of box being moved
        self.move_offset = None  # (dx, dy) from mouse to box center
        self.hover_idx = None  # index of box being hovered (for cursor feedback)
        # Error resilience
        self.failed_reads = 0
        self.last_frame = None  # Cache last good frame


def constrain_to_square(x0, y0, x, y):
    """Constrain rectangle to axis-aligned square."""
    dx, dy = x - x0, y - y0
    side = max(abs(dx), abs(dy))
    return x0, y0, x0 + (side if dx >= 0 else -side), y0 + (side if dy >= 0 else -side)


def get_box_center_rect(box):
    """Get the clickable center rectangle for a box (where the ID label is)."""
    x0, y0, x1, y1, _ = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    # Label area is roughly 20x20 pixels around center
    return cx - 15, cy - 15, cx + 15, cy + 15


def point_in_rect(x, y, rect):
    """Check if point (x,y) is inside rectangle (x0,y0,x1,y1)."""
    rx0, ry0, rx1, ry1 = rect
    return min(rx0, rx1) <= x <= max(rx0, rx1) and min(ry0, ry1) <= y <= max(ry0, ry1)


def find_box_at_center(state, x, y):
    """Find index of box whose center label contains point (x,y). Returns -1 if none."""
    # Check in reverse order so topmost (most recently added) box is selected first
    for i in range(len(state.boxes) - 1, -1, -1):
        if point_in_rect(x, y, get_box_center_rect(state.boxes[i])):
            return i
    return -1


def make_mouse_callback(state, window_name):
    """Create mouse callback for drawing and moving squares."""
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check if clicking on a box center to move it
            idx = find_box_at_center(state, x, y)
            if idx >= 0:
                box = state.boxes[idx]
                cx, cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                state.moving_idx = idx
                state.move_offset = (cx - x, cy - y)
            else:
                # Start drawing new box
                state.drawing = True
                state.start_pt = (x, y)
                state.current_box = None
                
        elif event == cv2.EVENT_MOUSEMOVE:
            # Update hover state for cursor feedback
            state.hover_idx = find_box_at_center(state, x, y) if state.moving_idx is None else state.moving_idx
            
            if state.moving_idx is not None and state.move_offset:
                # Move the box
                box = state.boxes[state.moving_idx]
                old_cx, old_cy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                new_cx = x + state.move_offset[0]
                new_cy = y + state.move_offset[1]
                dx, dy = new_cx - old_cx, new_cy - old_cy
                state.boxes[state.moving_idx] = [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy, box[4]]
            elif state.drawing and state.start_pt:
                state.current_box = constrain_to_square(*state.start_pt, x, y)
                
        elif event == cv2.EVENT_LBUTTONUP:
            if state.moving_idx is not None:
                box = state.boxes[state.moving_idx]
                print(f"[{window_name}] Moved box {box[4]} to ({box[0]},{box[1]}) -> ({box[2]},{box[3]})")
                state.moving_idx = None
                state.move_offset = None
            elif state.drawing and state.start_pt:
                x0, y0, x1, y1 = constrain_to_square(*state.start_pt, x, y)
                if abs(x1 - x0) > 5:  # minimum size
                    state.boxes.append([x0, y0, x1, y1, state.next_id])
                    print(f"[{window_name}] Box {state.next_id}: ({x0},{y0}) -> ({x1},{y1})")
                    state.next_id += 1
                state.drawing = False
                state.start_pt = None
                state.current_box = None
    return on_mouse


def draw_overlay(frame, state):
    """Draw boxes and IDs on frame."""
    out = frame.copy()
    
    # Draw finalized boxes
    for i, box in enumerate(state.boxes):
        x0, y0, x1, y1, box_id = box
        is_hovered = (state.hover_idx == i)
        is_moving = (state.moving_idx == i)
        
        # Box outline - cyan when being moved, green otherwise
        box_color = (255, 255, 0) if is_moving else (0, 255, 0)
        cv2.rectangle(out, (x0, y0), (x1, y1), box_color, 2)
        
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        text = str(box_id)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        
        # Label background - highlight when hovered/moving to show it's draggable
        if is_hovered or is_moving:
            # Draw move cursor indicator (4 arrows)
            bg_color = (80, 80, 80)
            label_color = (0, 255, 255)  # Cyan text when hoverable
            # Draw small arrows around the label to indicate "move"
            arrow_len = 8
            cv2.arrowedLine(out, (cx, cy - th//2 - 10), (cx, cy - th//2 - 10 - arrow_len), label_color, 1, tipLength=0.5)
            cv2.arrowedLine(out, (cx, cy + th//2 + 10), (cx, cy + th//2 + 10 + arrow_len), label_color, 1, tipLength=0.5)
            cv2.arrowedLine(out, (cx - tw//2 - 10, cy), (cx - tw//2 - 10 - arrow_len, cy), label_color, 1, tipLength=0.5)
            cv2.arrowedLine(out, (cx + tw//2 + 10, cy), (cx + tw//2 + 10 + arrow_len, cy), label_color, 1, tipLength=0.5)
        else:
            bg_color = (0, 0, 0)
            label_color = (0, 255, 0)
        
        cv2.rectangle(out, (cx - tw//2 - 4, cy - th//2 - 4), (cx + tw//2 + 4, cy + th//2 + 4), bg_color, -1)
        cv2.putText(out, text, (cx - tw//2, cy + th//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, label_color, 2, cv2.LINE_AA)
    
    # Draw preview box while dragging
    if state.drawing and state.current_box:
        cv2.rectangle(out, state.current_box[:2], state.current_box[2:], (0, 255, 255), 2)
    
    # Recording indicator
    if state.recording:
        cv2.circle(out, (30, 30), 10, (0, 0, 255), -1)
        cv2.putText(out, "REC", (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return out


def parse_cam_id(cam_str):
    """Parse camera identifier (path or index)."""
    if cam_str.startswith('/dev/'):
        return cam_str
    try:
        return f"/dev/video{int(cam_str)}"
    except ValueError:
        return cam_str


def load_annotations(path: Path, states: dict):
    """Load annotations from JSON file(s) into camera states."""
    files_to_load = []
    
    if path.is_file():
        files_to_load = [path]
    elif path.is_dir():
        files_to_load = list(path.glob("boxes_*.json"))
    
    loaded = 0
    for json_path in files_to_load:
        try:
            with open(json_path) as f:
                data = json.load(f)
            
            cam_id = data.get("camera")
            boxes = data.get("boxes", [])
            
            # Find matching state by camera ID
            if cam_id in states:
                state = states[cam_id]
                state.boxes = [[b["x0"], b["y0"], b["x1"], b["y1"], b["id"]] for b in boxes]
                state.next_id = max((b["id"] for b in boxes), default=0) + 1
                print(f"Loaded {len(boxes)} boxes for {cam_id} from {json_path}")
                loaded += 1
            else:
                # Try to match by partial camera ID (in case paths differ slightly)
                for state_cam_id, state in states.items():
                    if cam_id.replace('/', '_') in state_cam_id.replace('/', '_') or \
                       state_cam_id.replace('/', '_') in cam_id.replace('/', '_'):
                        state.boxes = [[b["x0"], b["y0"], b["x1"], b["y1"], b["id"]] for b in boxes]
                        state.next_id = max((b["id"] for b in boxes), default=0) + 1
                        print(f"Loaded {len(boxes)} boxes for {state_cam_id} from {json_path}")
                        loaded += 1
                        break
                else:
                    print(f"No matching camera for {cam_id} in {json_path}")
        except Exception as e:
            print(f"Failed to load {json_path}: {e}")
    
    return loaded


def main():
    parser = argparse.ArgumentParser(description="Draw numbered squares on camera feeds")
    parser.add_argument('--cams', nargs='+', default=['/dev/video0'], help='Camera paths or indices')
    parser.add_argument('--width', type=int, default=1280, help='Frame width')
    parser.add_argument('--height', type=int, default=720, help='Frame height')
    parser.add_argument('--fps', type=float, default=30.0, help='Frame rate')
    parser.add_argument('--record-output', type=str, help='Output video path (use {cam} placeholder)')
    parser.add_argument('--load', type=str, default='outputs/annotations',
                       help='Load saved annotations from file or directory (default: outputs/annotations)')
    args = parser.parse_args()

    print("Keys: q=quit, c=clear, u=undo, r=record, s=snapshot")
    print("Mouse: click+drag to draw, drag center label to move\n")

    # Initialize cameras
    cameras, states = {}, {}
    for cam_str in args.cams:
        cam_id = parse_cam_id(cam_str)
        config = OpenCVCameraConfig(
            index_or_path=cam_id, fps=args.fps, width=args.width, height=args.height,
            color_mode=ColorMode.RGB, rotation=Cv2Rotation.NO_ROTATION, fourcc="MJPG"
        )
        camera = OpenCVCamera(config)
        try:
            camera.connect()
            print(f"Connected: {cam_id}")
            cameras[cam_id] = camera
            states[cam_id] = CameraState(cam_id)
            window = f"Cam: {cam_id}"
            cv2.namedWindow(window)
            cv2.setMouseCallback(window, make_mouse_callback(states[cam_id], window))
        except Exception as e:
            print(f"Failed: {cam_id} - {e}")

    if not cameras:
        print("No cameras available!")
        sys.exit(1)

    # Load saved annotations if available
    load_path = Path(args.load)
    if load_path.exists():
        loaded = load_annotations(load_path, states)
        if loaded > 0:
            print(f"Resumed {loaded} annotation(s)\n")
    else:
        print(f"No saved annotations at {load_path}\n")

    # Main loop - resilient to temporary camera issues
    try:
        while True:  # Don't exit even if cameras temporarily fail
            for cam_id in list(cameras.keys()):
                camera, state = cameras[cam_id], states[cam_id]
                window = f"Cam: {cam_id}"
                
                try:
                    frame = camera.async_read(timeout_ms=100)  # Shorter timeout for responsiveness
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    state.last_frame = frame_bgr  # Cache good frame
                    state.failed_reads = 0
                    annotated = draw_overlay(frame_bgr, state)
                    cv2.imshow(window, annotated)
                    
                    # Recording
                    if state.recording and state.writer:
                        state.writer.write(annotated)
                except Exception:
                    state.failed_reads += 1
                    # Show last good frame with error overlay if available
                    if state.last_frame is not None:
                        annotated = draw_overlay(state.last_frame.copy(), state)
                        cv2.putText(annotated, f"Camera read error ({state.failed_reads})", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        cv2.imshow(window, annotated)
                    # Only disconnect after many consecutive failures
                    if state.failed_reads > 300:  # ~10 seconds at 30fps
                        print(f"[{cam_id}] Too many failures, disconnecting")
                        camera.disconnect()
                        del cameras[cam_id]

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == ord('c'):
                for s in states.values():
                    s.boxes.clear()
                    s.next_id = 1
                print("Cleared all boxes")
            elif key == ord('u'):
                undone = False
                for s in states.values():
                    if s.boxes:
                        s.boxes.pop()
                        s.next_id = max(1, s.next_id - 1)
                        undone = True
                if undone:
                    print("Undid last box on all windows")
            elif key == ord('r') and args.record_output:
                for cam_id, s in states.items():
                    s.recording = not s.recording
                    if s.recording and s.writer is None:
                        safe = cam_id.replace('/', '_')
                        path = args.record_output.replace('{cam}', safe)
                        Path(path).parent.mkdir(parents=True, exist_ok=True)
                        s.writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'MJPG'), 
                                                   args.fps, (args.width, args.height))
                        print(f"Recording: {path}")
                    elif not s.recording:
                        print("Recording stopped")
            elif key == ord('s'):
                Path("outputs/snapshots").mkdir(parents=True, exist_ok=True)
                ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                for cam_id, state in states.items():
                    # Use cached frame if camera read fails
                    frame_bgr = None
                    if cam_id in cameras:
                        try:
                            frame = cameras[cam_id].async_read(timeout_ms=100)
                            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        except Exception:
                            pass
                    if frame_bgr is None and state.last_frame is not None:
                        frame_bgr = state.last_frame.copy()
                    if frame_bgr is not None:
                        annotated = draw_overlay(frame_bgr, state)
                        path = f"outputs/snapshots/snapshot_{cam_id.replace('/', '_')}_{ts}.png"
                        cv2.imwrite(path, annotated)
                        print(f"Saved: {path}")

    except KeyboardInterrupt:
        pass

    finally:
        # Cleanup
        for s in states.values():
            if s.writer:
                s.writer.release()
        for camera in cameras.values():
            camera.disconnect()
        cv2.destroyAllWindows()

        # Save annotations
        has_boxes = any(s.boxes for s in states.values())
        if has_boxes:
            out_dir = Path('outputs/annotations')
            out_dir.mkdir(parents=True, exist_ok=True)
            for cam_id, s in states.items():
                data = {"camera": cam_id, "frame_width": args.width, "frame_height": args.height,
                        "boxes": [{"id": i, "x0": x0, "y0": y0, "x1": x1, "y1": y1} 
                                  for x0, y0, x1, y1, i in s.boxes]}
                path = out_dir / f"boxes_{cam_id.replace('/', '_')}.json"
                with open(path, 'w') as f:
                    json.dump(data, f, indent=2)
                print(f"Saved: {path}")


if __name__ == "__main__":
    main()
