import os

import cv2


def check_and_record():
    # --- CẤU HÌNH INDEX ---
    # Thường 0 là Wrist Cam, 1 là Phone Cam (DroidCam)
    cam_index_wrist = 0
    cam_index_front = 3

    # Thư mục lưu test
    save_dir = "camera_test"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    # Khởi tạo camera
    cap_wrist = cv2.VideoCapture(cam_index_wrist)
    cap_front = cv2.VideoCapture(cam_index_front)

    # Thử set độ phân giải mong muốn
    cap_wrist.set(cv2.CAP_PROP_FRAME_WIDTH, 800)
    cap_wrist.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
    cap_front.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap_front.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def get_info(cap, name):
        # FIX: đổi is_opened() thành isOpened()
        if not cap.isOpened():
            return f"❌ {name} không mở được!"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        return f"✅ {name}: {w}x{h} @ {fps} FPS"

    print("-" * 30)
    print(get_info(cap_wrist, "Wrist Camera"))
    print(get_info(cap_front, "Front Camera (Phone)"))
    print("-" * 30)

    # Thiết lập lưu video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    # Lấy size thực tế để init VideoWriter cho đúng
    w_w, h_w = int(cap_wrist.get(3)), int(cap_wrist.get(4))
    w_f, h_f = int(cap_front.get(3)), int(cap_front.get(4))

    out_wrist = None
    if cap_wrist.isOpened():
        out_wrist = cv2.VideoWriter(f"{save_dir}/test_wrist.mp4", fourcc, 20.0, (w_w, h_w))

    out_front = None
    if cap_front.isOpened():
        out_front = cv2.VideoWriter(f"{save_dir}/test_front.mp4", fourcc, 20.0, (w_f, h_f))

    print("🎥 Đang hiển thị và ghi nháp... Nhấn 'q' để dừng và lưu.")

    while True:
        ret_w, frame_w = cap_wrist.read() if cap_wrist.isOpened() else (False, None)
        ret_f, frame_f = cap_front.read() if cap_front.isOpened() else (False, None)

        if ret_w:
            # Resize để hiển thị (không ảnh hưởng đến file lưu)
            view_w = cv2.resize(frame_w, (800, 600))
            cv2.imshow("Wrist View (Preview)", view_w)
            if out_wrist:
                out_wrist.write(frame_w)

        if ret_f:
            # FIX: Resize ảnh 2K xuống để nhìn được toàn cảnh trên màn hình Laptop
            # 640x360 là tỉ lệ 16:9 chuẩn, giúp bạn thấy hết cái bàn
            view_f = cv2.resize(frame_f, (1280, 720))
            cv2.imshow("Front View (Preview)", view_f)
            if out_front:
                out_front.write(frame_f)

        if (cv2.waitKey(1) & 0xFF == ord("q")) or (not ret_w and not ret_f):
            break

    # Giải phóng
    cap_wrist.release()
    cap_front.release()
    if out_wrist:
        out_wrist.release()
    if out_front:
        out_front.release()
    cv2.destroyAllWindows()
    print(f"✅ Đã lưu video test vào thư mục '{save_dir}'")


if __name__ == "__main__":
    check_and_record()
