import cv2
import tensorflow as tf
import numpy as np

interpreter = tf.lite.Interpreter(model_path="models/movenet_thunder_singlepose.tflite")
interpreter.allocate_tensors()

INPUT_DETAILS = interpreter.get_input_details()
OUTPUT_DETAILS = interpreter.get_output_details()
model_size = 256


def movenet_pose_detection(input_video, output_video):
    print("MoveNet: Start processing video")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("MoveNet: Cant open video file")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video+"_movenet.mp4", fourcc, fps, (frame_width, frame_height))

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

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (model_size, model_size))
        image = np.expand_dims(image.astype(np.float32) / 255.0, axis=0)

        keypoints_rel = np.squeeze(keypoints_rel)
        keypoints = keypoints_rel[:, :2] * [model_size, model_size]

        scale = max(orig_height, orig_width) / model_size
        pad_y = (model_size - orig_height / scale) / 2
        pad_x = (model_size - orig_width / scale) / 2

    
        keypoints[:, 0] = (keypoints[:, 0] - pad_y) * scale
        keypoints[:, 1] = (keypoints[:, 1] - pad_x) * scale

        adjusted_keypoints = np.hstack([keypoints, keypoints_rel[:, 2:3]])

        for kp in adjusted_keypoints:
            y, x, confidence = kp

            if confidence > 0.5:
                cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("MoveNet: Video done: ", output_video)