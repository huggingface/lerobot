import cv2


def discover_cameras(max_to_test=5):
    print("🔍 Đang quét các cổng camera...")
    for i in range(max_to_test):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Đọc 1 frame để chắc chắn cam hoạt động
            ret, frame = cap.read()
            if ret:
                window_name = f"Camera Index {i}"
                cv2.imshow(window_name, frame)
                print(f"✅ Tìm thấy camera tại Index: {i}")
                print("   Nhấn phím bất kỳ để xem camera tiếp theo...")
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)
            cap.release()
        else:
            print(f"❌ Index {i}: Không có thiết bị.")

    cv2.destroyAllWindows()
    print("\n🏁 Đã quét xong. Hãy ghi lại số Index của Wrist Cam và Phone Cam.")


if __name__ == "__main__":
    discover_cameras()
