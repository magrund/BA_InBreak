import cv2
import numpy as np
from keypoints_and_pairs import BODY_25_KEYPOINTS, BODY_25_KEYPOINTS_PAIRS

proto_file = "models/body25/pose_deploy.prototxt"
weights_file = "models/body25/pose_iter_584000.caffemodel"

n_points = len(BODY_25_KEYPOINTS)

net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

color = (0, 255, 255)
confidence_threshold = 0.1

def openpose_body25_pose_detection(input_video, output_video):
    print("OpenPose BODY25: Start processing video")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("OpenPose BODY25: Can't open video file")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video + "_openpose_body25.mp4", fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]
        inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inp_blob)
        output = net.forward()

        points = {}

        for idx, keypoint in enumerate(BODY_25_KEYPOINTS):
            heatmap = output[0, keypoint, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatmap)

            x = int(frame_width * point[0] / output.shape[3])
            y = int(frame_height * point[1] / output.shape[2])

            if conf > confidence_threshold:
                points[keypoint] = (x, y)
                cv2.circle(frame, (x, y), 5, color, thickness=-1, lineType=cv2.FILLED)

        for part_a, part_b in BODY_25_KEYPOINTS_PAIRS:
            if part_a in points and part_b in points:
                cv2.line(frame, points[part_a], points[part_b], color, 2, lineType=cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("OpenPose BODY25: Video done:", output_video)
