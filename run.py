import cv2
from tfdat import Tfdat

stream_url = "rtsp://cs2projs:cs2projs@192.168.0.166/stream2"
class_names = [
    "Bicycle",
    "Chair",
    "Box",
    "Table",
    "Plastic bag",
    "Flowerpot",
    "Luggage and bags",
    "Umbrella",
    "Shopping trolley",
    "Person",
]
while True:

    detect = Tfdat("yolov8_hdb.tflite")
    track = detect.track_video(stream_url, class_names=class_names)
    counter = 0
    total_fps = 0
    for bbox_details, frame_details in track:
        frame, frame_num, fps = frame_details
        counter += 1
        total_fps += fps
        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
        cv2.imshow("Live Video", frame)

    print(f"Avg fps: {total_fps/counter}")