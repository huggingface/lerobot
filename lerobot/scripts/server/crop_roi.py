import cv2

from lerobot.common.robot_devices.cameras.opencv import OpenCVCamera


def select_square_roi(img):
    """
    Allows the user to draw a square ROI on the image.

    The user must click and drag to draw the square.
    - While dragging, the square is dynamically drawn.
    - On mouse button release, the square is fixed.
    - Press 'c' to confirm the selection.
    - Press 'r' to reset the selection.
    - Press ESC to cancel.

    Returns:
        A tuple (top, left, height, width) representing the square ROI,
        or None if no valid ROI is selected.
    """
    # Create a working copy of the image
    clone = img.copy()
    working_img = clone.copy()

    roi = None  # Will store the final ROI as (top, left, side, side)
    drawing = False
    ix, iy = -1, -1  # Initial click coordinates

    def mouse_callback(event, x, y, flags, param):
        nonlocal ix, iy, drawing, roi, working_img

        if event == cv2.EVENT_LBUTTONDOWN:
            # Start drawing: record starting coordinates
            drawing = True
            ix, iy = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                # Compute side length as the minimum of horizontal/vertical drags
                side = min(abs(x - ix), abs(y - iy))
                # Determine the direction to draw (in case of dragging to top/left)
                dx = side if x >= ix else -side
                dy = side if y >= iy else -side
                # Show a temporary image with the current square drawn
                temp = working_img.copy()
                cv2.rectangle(temp, (ix, iy), (ix + dx, iy + dy), (0, 255, 0), 2)
                cv2.imshow("Select ROI", temp)

        elif event == cv2.EVENT_LBUTTONUP:
            # Finish drawing
            drawing = False
            side = min(abs(x - ix), abs(y - iy))
            dx = side if x >= ix else -side
            dy = side if y >= iy else -side
            # Normalize coordinates: (top, left) is the minimum of the two points
            x1 = min(ix, ix + dx)
            y1 = min(iy, iy + dy)
            roi = (y1, x1, side, side)  # (top, left, height, width)
            # Draw the final square on the working image and display it
            working_img = clone.copy()
            cv2.rectangle(working_img, (ix, iy), (ix + dx, iy + dy), (0, 255, 0), 2)
            cv2.imshow("Select ROI", working_img)

    # Create the window and set the callback
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", mouse_callback)
    cv2.imshow("Select ROI", working_img)

    print("Instructions for ROI selection:")
    print("  - Click and drag to draw a square ROI.")
    print("  - Press 'c' to confirm the selection.")
    print("  - Press 'r' to reset and draw again.")
    print("  - Press ESC to cancel the selection.")

    # Wait until the user confirms with 'c', resets with 'r', or cancels with ESC
    while True:
        key = cv2.waitKey(1) & 0xFF
        # Confirm ROI if one has been drawn
        if key == ord("c") and roi is not None:
            break
        # Reset: clear the ROI and restore the original image
        elif key == ord("r"):
            working_img = clone.copy()
            roi = None
            cv2.imshow("Select ROI", working_img)
        # Cancel selection for this image
        elif key == 27:  # ESC key
            roi = None
            break

    cv2.destroyWindow("Select ROI")
    return roi


def select_square_roi_for_images(images: dict) -> dict:
    """
    For each image in the provided dictionary, open a window to allow the user
    to select a square ROI. Returns a dictionary mapping each key to a tuple
    (top, left, height, width) representing the ROI.

    Parameters:
        images (dict): Dictionary where keys are identifiers and values are OpenCV images.

    Returns:
        dict: Mapping of image keys to the selected square ROI.
    """
    selected_rois = {}

    for key, img in images.items():
        if img is None:
            print(f"Image for key '{key}' is None, skipping.")
            continue

        print(f"\nSelect square ROI for image with key: '{key}'")
        roi = select_square_roi(img)

        if roi is None:
            print(f"No valid ROI selected for '{key}'.")
        else:
            selected_rois[key] = roi
            print(f"ROI for '{key}': {roi}")

    return selected_rois


if __name__ == "__main__":
    # Example usage:
    # Replace 'image1.jpg' and 'image2.jpg' with valid paths to your image files.
    fps = [5, 30]
    cameras = [OpenCVCamera(i, fps=fps[i], width=640, height=480, mock=False) for i in range(2)]
    [camera.connect() for camera in cameras]

    image_keys = ["image_" + str(i) for i in range(len(cameras))]

    images = {image_keys[i]: cameras[i].read() for i in range(len(cameras))}

    # Verify images loaded correctly
    for key, img in images.items():
        if img is None:
            raise ValueError(f"Failed to load image for key '{key}'. Check the file path.")

    # Let the user select a square ROI for each image
    rois = select_square_roi_for_images(images)

    # Print the selected square ROIs
    print("\nSelected Square Regions of Interest (top, left, height, width):")
    for key, roi in rois.items():
        print(f"{key}: {roi}")
