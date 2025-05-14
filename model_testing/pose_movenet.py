import cv2
import tensorflow as tf
import numpy as np
from keypoints_and_pairs import COCO_KEYPOINT_PAIRS
from colors import movenet_color

interpreter = tf.lite.Interpreter(model_path="MODEL_PATH")
interpreter.allocate_tensors()

INPUT_DETAILS = interpreter.get_input_details()
OUTPUT_DETAILS = interpreter.get_output_details()
model_size = "INT NUMBER"

confidence_threshold = 0.5

def movenet_pose_detection(input_video, output_video):
    print("MoveNet: Start processing video")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("MoveNet: Cant open video file")
        return None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video+"_movenet.mp4", fourcc, fps, (frame_width, frame_height))

    keypoints_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        orig_height, orig_width, _ = frame.shape
        
        image = tf.image.resize_with_pad(np.expand_dims(frame, axis=0), model_size, model_size)
        input_image = tf.cast(image, dtype=tf.float32)

        interpreter.set_tensor(INPUT_DETAILS[0]['index'], input_image.numpy())
        interpreter.invoke()
        keypoints_rel = interpreter.get_tensor(OUTPUT_DETAILS[0]['index'])

        keypoints_rel = np.squeeze(keypoints_rel)
        keypoints = keypoints_rel[:, :2] * [model_size, model_size]

        scale = max(orig_height, orig_width) / model_size
        pad_y = (model_size - orig_height / scale) / 2
        pad_x = (model_size - orig_width / scale) / 2

        keypoints[:, 0] = (keypoints[:, 0] - pad_y) * scale
        keypoints[:, 1] = (keypoints[:, 1] - pad_x) * scale

        adjusted_keypoints = np.hstack([keypoints, keypoints_rel[:, 2:3]])

        frame_keypoints = []

        for kp in adjusted_keypoints:
            y, x, confidence = kp
            frame_keypoints.append({"x": int(x), "y": int(y), "confidence": confidence})
            if confidence > confidence_threshold:
                cv2.circle(frame, (int(x), int(y)), 5, movenet_color, -1)

        keypoints_list.append(frame_keypoints)

        for pair in COCO_KEYPOINT_PAIRS:
            kp1, kp2 = adjusted_keypoints[pair[0]], adjusted_keypoints[pair[1]]
            y1, x1, conf1 = kp1
            y2, x2, conf2 = kp2
            if conf1 > confidence_threshold and conf2 > confidence_threshold:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), movenet_color, 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("MoveNet: Video done: ", output_video)

    return keypoints_list