import cv2
from ultralytics import YOLO
import numpy as np
from keypoints_and_pairs import COCO_KEYPOINT_PAIRS

model = YOLO("models/yolo11m-pose.pt")
color = (255, 0, 0)
confidence_threshold = 0.5

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
            kpts = result.keypoints.data[0].cpu().numpy()

            for kpt in kpts:
                x, y, conf = kpt
                if conf > confidence_threshold:
                    cv2.circle(frame, (int(x), int(y)), 5, color, -1)

            for pair in COCO_KEYPOINT_PAIRS:
                if pair[0] < len(kpts) and pair[1] < len(kpts):
                    x1, y1, conf1 = kpts[pair[0]]
                    x2, y2, conf2 = kpts[pair[1]]
                    if conf1 > confidence_threshold and conf2 > confidence_threshold:
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("YOLO: Video done: ", output_video)
