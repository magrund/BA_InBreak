import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("models/yolo11m-pose.pt")

def yolo_pose_detection(input_video, output_video):
    print("YOLO: Start processing video")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("YOLO: Cant open video file")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video+"_yolo.mp4", fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        for result in results:
            kpts = result.keypoints.data
            for kpt in kpts[0]:
                x, y, conf = kpt.tolist()
                if conf > 0.5:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("YOLO: Video done: ", output_video)