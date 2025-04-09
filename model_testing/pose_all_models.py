import cv2
from keypoints_and_pairs import COCO_KEYPOINT_PAIRS, OPENPOSE_BODY_25_KEYPOINTS_PAIRS, OPENPOSE_COCO_KEYPOINTS_PAIRS
from colors import movenet_color, yolo_color, mediapipe_with_tracking_color, mediapipe_no_tracking_color, openpose_coco_color, openpose_body25_color

def all_models_pose_detection(input_video, output_video, movenet_keypoints, yolo_keypoints, mediapipe_with_tracking_keypoints, mediapipe_no_tracking_keypoints, openpose_coco_keypoints, openpose_body25_keypoints):
    confidence_treshhold = 0.1
    print("All models: Start processing video")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("All models: Cant open video file")
        return None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video + "_all_models.mp4", fourcc, fps, (frame_width, frame_height))

    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        """
        if frame_idx < len(movenet_keypoints):
            # MoveNet
            for kp in movenet_keypoints[frame_idx]:  
                x, y, confidence = kp["x"], kp["y"], kp["confidence"]

                if confidence >= confidence_treshhold:
                    cv2.circle(frame, (int(x), int(y)), 5, movenet_color, -1)

            for pair in COCO_KEYPOINT_PAIRS:
                if frame_idx < len(movenet_keypoints) and len(movenet_keypoints[frame_idx]) > max(pair[0], pair[1]):
                    kp1 = movenet_keypoints[frame_idx][pair[0]]
                    kp2 = movenet_keypoints[frame_idx][pair[1]]

                    if kp1 and kp2 and kp1["confidence"]>=confidence_treshhold and kp2["confidence"]>=confidence_treshhold:
                        x1, y1 = kp1["x"], kp1["y"]
                        x2, y2 = kp2["x"], kp2["y"]
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), movenet_color, 2)

        if frame_idx < len(yolo_keypoints):
            # YOLO
            for kp in yolo_keypoints[frame_idx]:  
                x, y, confidence = kp["x"], kp["y"], kp["confidence"]

                if confidence >= confidence_treshhold:
                    cv2.circle(frame, (int(x), int(y)), 5, yolo_color, -1)

            for pair in COCO_KEYPOINT_PAIRS:
                if frame_idx < len(yolo_keypoints) and len(yolo_keypoints[frame_idx]) > max(pair[0], pair[1]):
                    kp1 = yolo_keypoints[frame_idx][pair[0]]
                    kp2 = yolo_keypoints[frame_idx][pair[1]]

                    if kp1 and kp2 and kp1["x"]>0 and kp1["y"]>0 and kp2["x"]>0 and kp2["y"]>0:
                        x1, y1 = kp1["x"], kp1["y"]
                        x2, y2 = kp2["x"], kp2["y"]
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), yolo_color, 2)

        if frame_idx < len(mediapipe_with_tracking_keypoints):
            # MediaPipe with Tracking
            for kp in mediapipe_with_tracking_keypoints[frame_idx]:  
                x, y, confidence = kp["x"], kp["y"], kp["confidence"]
                if confidence >= confidence_treshhold:
                    cv2.circle(frame, (int(x), int(y)), 5, mediapipe_with_tracking_color, -1)

            for pair in COCO_KEYPOINT_PAIRS:
                if frame_idx < len(mediapipe_with_tracking_keypoints) and len(mediapipe_with_tracking_keypoints[frame_idx]) > max(pair[0], pair[1]):
                    kp1 = mediapipe_with_tracking_keypoints[frame_idx][pair[0]]
                    kp2 = mediapipe_with_tracking_keypoints[frame_idx][pair[1]]

                    if kp1 and kp2 and kp1["confidence"]>=confidence_treshhold and kp2["confidence"]>=confidence_treshhold:
                        x1, y1 = kp1["x"], kp1["y"]
                        x2, y2 = kp2["x"], kp2["y"]
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), mediapipe_with_tracking_color, 2)

        if frame_idx < len(mediapipe_no_tracking_keypoints):
            # MediaPipe without Tracking
            for kp in mediapipe_no_tracking_keypoints[frame_idx]:  
                x, y, confidence = kp["x"], kp["y"], kp["confidence"]

                if confidence >= confidence_treshhold:
                    cv2.circle(frame, (int(x), int(y)), 5, mediapipe_no_tracking_color, -1)

            for pair in COCO_KEYPOINT_PAIRS:
                if frame_idx < len(mediapipe_no_tracking_keypoints) and len(mediapipe_no_tracking_keypoints[frame_idx]) > max(pair[0], pair[1]):
                    kp1 = mediapipe_no_tracking_keypoints[frame_idx][pair[0]]
                    kp2 = mediapipe_no_tracking_keypoints[frame_idx][pair[1]]

                    if kp1 and kp2 and kp1["confidence"]>=confidence_treshhold and kp2["confidence"]>=confidence_treshhold:
                        x1, y1 = kp1["x"], kp1["y"]
                        x2, y2 = kp2["x"], kp2["y"]
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), mediapipe_no_tracking_color, 2)

        if frame_idx < len(openpose_coco_keypoints):
            # OpenPose COCO
            for kp in openpose_coco_keypoints[frame_idx]:  
                x, y, confidence = kp["x"], kp["y"], kp["confidence"]
                
                if confidence >= confidence_treshhold:
                    cv2.circle(frame, (int(x), int(y)), 5, openpose_coco_color, -1)

            for pair in COCO_KEYPOINT_PAIRS:
                if frame_idx < len(openpose_coco_keypoints) and len(openpose_coco_keypoints[frame_idx]) > max(pair[0], pair[1]):
                    kp1 = openpose_coco_keypoints[frame_idx][pair[0]]
                    kp2 = openpose_coco_keypoints[frame_idx][pair[1]]

                    if kp1 and kp2 and kp1["confidence"]>=confidence_treshhold and kp2["confidence"]>=confidence_treshhold:
                        x1, y1 = kp1["x"], kp1["y"]
                        x2, y2 = kp2["x"], kp2["y"]
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), openpose_coco_color, 2)
        """
        if frame_idx < len(openpose_body25_keypoints):
            # OpenPose Body25
            for kp in openpose_body25_keypoints[frame_idx]:  
                x, y, confidence, idx = kp["x"], kp["y"], kp["confidence"], kp["idx"]

                if confidence >= confidence_treshhold:
                    cv2.circle(frame, (int(x), int(y)), 5, openpose_body25_color, -1)
                    cv2.putText(frame, str(idx), (int(x) + 10, int(y) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, openpose_body25_color, 2)
            """
            for pair in OPENPOSE_BODY_25_KEYPOINTS_PAIRS:
                if frame_idx < len(openpose_body25_keypoints) and len(openpose_body25_keypoints[frame_idx]) > max(pair[0], pair[1]):
                    kp1 = openpose_body25_keypoints[frame_idx][pair[0]]
                    kp2 = openpose_body25_keypoints[frame_idx][pair[1]]

                    if kp1 and kp2 and kp1["confidence"]>=confidence_treshhold and kp2["confidence"]>=confidence_treshhold:
                        x1, y1 = kp1["x"], kp1["y"]
                        x2, y2 = kp2["x"], kp2["y"]
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), openpose_body25_color, 2)
            """
    
        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("All models: Video saved as", output_video + "_all_models.mp4")