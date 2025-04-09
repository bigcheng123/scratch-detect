import cv2
import threading

def enumerate_cameras():
    # 获取所有可用的相机设备
    index = 0
    cameras = []
    while True:
        cap = cv2.VideoCapture(index)
        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3968)  # 设置图像宽度
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2976)  # 设置图像高度
        # cap.set(cv2.CAP_PROP_FPS, 6)  # 设置帧率
        if not cap.isOpened():
            break
        else:
            cameras.append(index)
            cap.release()
        index += 1
    return cameras

def open_camera(camera_index):
    cap = cv2.VideoCapture(camera_index)
    while True:
        ret, frame = cap.read()


        if not ret:
            break
        cv2.imshow(f"Camera {camera_index}", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    cap.release()

if __name__ == "__main__":
    cameras = enumerate_cameras()
    if len(cameras) > 0:
        print("Available cameras:")
        for camera in cameras:
            print(f"Camera {camera}")
            # threading.Thread(target=open_camera, args=(camera,)).start() #开启摄像头
    else:
        print("No cameras found.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
