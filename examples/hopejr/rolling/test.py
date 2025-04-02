import numpy as np
from PIL import Image, ImageSequence

def coalesce_gif(im):
    """
    Attempt to coalesce frames so each one is a full image.
    This handles many (though not all) partial-frame GIFs.
    """
    # Convert mode to RGBA
    im = im.convert("RGBA")

    # Prepare an accumulator the same size as the base frame
    base = Image.new("RGBA", im.size)
    frames = []
    
    # Go through each frame
    for frame in ImageSequence.Iterator(im):
        base.alpha_composite(frame.convert("RGBA"))
        frames.append(base.copy())
    return frames

def remove_white_make_black(arr, threshold=250):
    """
    For each pixel in arr (H,W,3), if R,G,B >= threshold, set to black (0,0,0).
    This effectively 'removes' white so it won't affect the sum.
    """
    mask = (arr[..., 0] >= threshold) & \
           (arr[..., 1] >= threshold) & \
           (arr[..., 2] >= threshold)
    arr[mask] = 0  # set to black

def main():
    # Load the animated GIF
    gif = Image.open("input.gif")
    
    # Coalesce frames so each is full-size
    frames = coalesce_gif(gif)
    if not frames:
        print("No frames found!")
        return
    
    # Convert first frame to RGB array, initialize sum array
    w, h = frames[0].size
    sum_array = np.zeros((h, w, 3), dtype=np.uint16)  # 16-bit to avoid overflow

    # For each frame:
    for f in frames:
        # Convert to RGB
        rgb = f.convert("RGB")
        arr = np.array(rgb, dtype=np.uint16)  # shape (H, W, 3)
        
        # Remove near-white by setting it to black
        remove_white_make_black(arr, threshold=250)
        
        # Add to sum_array, then clamp to 255
        sum_array += arr
        np.clip(sum_array, 0, 255, out=sum_array)
    
    # Convert sum_array back to 8-bit
    sum_array = sum_array.astype(np.uint8)

    # Finally, any pixel that stayed black is presumably "empty," so we set it to white
    black_mask = (sum_array[..., 0] == 0) & \
                 (sum_array[..., 1] == 0) & \
                 (sum_array[..., 2] == 0)
    sum_array[black_mask] = [255, 255, 255]

    # Create final Pillow image
    final_img = Image.fromarray(sum_array, mode="RGB")
    final_img.save("result.png")
    print("Done! Wrote result.png.")

if __name__ == "__main__":
    main()
