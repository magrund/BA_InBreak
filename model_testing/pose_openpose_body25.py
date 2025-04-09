import cv2
import numpy as np
from keypoints_and_pairs import BODY_25_KEYPOINTS, OPENPOSE_BODY_25_KEYPOINTS_PAIRS
from colors import openpose_body25_color

proto_file = "models/body25/pose_deploy.prototxt"
weights_file = "models/body25/pose_iter_584000.caffemodel"

n_points = len(BODY_25_KEYPOINTS)

net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

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

    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]
        inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inp_blob)
        output = net.forward()

        points = {}

        frame_keypoints = []

        for idx, keypoint in enumerate(BODY_25_KEYPOINTS):
            heatmap = output[0, keypoint, :, :]
            _, confidence, _, point = cv2.minMaxLoc(heatmap)

            x = int(frame_width * point[0] / output.shape[3])
            y = int(frame_height * point[1] / output.shape[2])

            frame_keypoints.append({"x": int(x), "y": int(y), "confidence": confidence, "idx": str(idx)})
            print(idx)

            if confidence > confidence_threshold:
                points[keypoint] = (x, y)
                cv2.circle(frame, (x, y), 5, openpose_body25_color, thickness=-1, lineType=cv2.FILLED)

        keypoints_list.append(frame_keypoints)

        for part_a, part_b in OPENPOSE_BODY_25_KEYPOINTS_PAIRS:
            if part_a in points and part_b in points:
                cv2.line(frame, points[part_a], points[part_b], openpose_body25_color, 2, lineType=cv2.LINE_AA)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("OpenPose BODY25: Video done:", output_video)

    return keypoints_list
