import cv2
import mediapipe as mp
from keypoints_and_pairs import MEDIAPIPE_KEYPOINTS, MEDIAPIPE_KEYPOINTS_PAIRS
from colors import mediapipe_no_tracking_color

confidence_threshold = 0.5

def mediapipe_no_tracking_pose_detection(input_video, output_video):
    print("MediaPipe no Tracking: Start processing video")
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("MediaPipe no Tracking: Cant open video file")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video+"_mediapipe_no_tracking.mp4", fourcc, fps, (frame_width, frame_height))

    keypoints_list = []

    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            frame_keypoints = []

            if results.pose_landmarks:
                keypoints = [None] * 33
                for idx in MEDIAPIPE_KEYPOINTS:
                    lm = results.pose_landmarks.landmark[idx]
                    frame_keypoints.append({"x": int(lm.x * frame_width), "y": int(lm.y * frame_height), "confidence": lm.visibility})
                    if lm.visibility > confidence_threshold:
                        cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)
                        keypoints[idx] = (cx, cy)
                        cv2.circle(image, (cx, cy), 5, mediapipe_no_tracking_color, -1)

                keypoints_list.append(frame_keypoints)

                for pair in MEDIAPIPE_KEYPOINTS_PAIRS:
                    if keypoints[pair[0]] and keypoints[pair[1]]:
                        cv2.line(image, keypoints[pair[0]], keypoints[pair[1]], mediapipe_no_tracking_color, 2)

            out.write(image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("MediaPipe no Tracking: Video done: ", output_video)

    return keypoints_list
