import cv2
import mediapipe as mp
from keypoints_and_pairs import MEDIAPIPE_KEYPOINTS, MEDIAPIPE_KEYPOINTS_PAIRS

color = (0, 0, 255)
confidence_threshold = 0.5

def mediapipe_pose_detection(input_video, output_video):
    print("MediaPipe: Start processing video")
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("MediaPipe: Cant open video file")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video+"_mediapipe.mp4", fourcc, fps, (frame_width, frame_height))

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if results.pose_landmarks:
                keypoints = [None] * 33
                for idx in MEDIAPIPE_KEYPOINTS:
                    lm = results.pose_landmarks.landmark[idx]
                    if lm.visibility > confidence_threshold:
                        cx, cy = int(lm.x * frame_width), int(lm.y * frame_height)
                        keypoints[idx] = (cx, cy)
                        cv2.circle(image, (cx, cy), 5, color, -1)

                for pair in MEDIAPIPE_KEYPOINTS_PAIRS:
                    if keypoints[pair[0]] and keypoints[pair[1]]:
                        cv2.line(image, keypoints[pair[0]], keypoints[pair[1]], color, 2)

            out.write(image)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("MediaPipe: Video done: ", output_video)
