import glob

import cv2


def make_video(folder, output_name, fps=30):
    images = sorted(glob.glob(f"{folder}/*.jpg"))
    if not images:
        return

    frame = cv2.imread(images[0])
    h, w, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_name, fourcc, fps, (w, h))

    print(f"🎬 Đang nén video: {output_name}...")
    for img_path in images:
        out.write(cv2.imread(img_path))
    out.release()
    print("✅ Xong!")


# Gộp cho cả 2 cam
make_video("teleop_recordings/wrist", "wrist_final.mp4", fps=30)
make_video("teleop_recordings/portal", "portal_final.mp4", fps=30)
