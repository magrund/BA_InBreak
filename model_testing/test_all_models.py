from pose_mediapipe_with_tracking import mediapipe_with_tracking_pose_detection
from pose_mediapipe_no_tracking import mediapipe_no_tracking_pose_detection
from pose_movenet import movenet_pose_detection
from pose_yolo import yolo_pose_detection
from pose_openpose_coco import openpose_coco_pose_detection
from pose_openpose_body25 import openpose_body25_pose_detection

input_video = 'INPUT_VIDEO_PATH'
output_video = 'OUTPUT_VIDEO_PATH'

if __name__ == '__main__':
    mediapipe_with_tracking_pose_detection(input_video, output_video)
    mediapipe_no_tracking_pose_detection(input_video, output_video)
    movenet_pose_detection(input_video, output_video)
    yolo_pose_detection(input_video, output_video)
    openpose_coco_pose_detection(input_video, output_video)
    openpose_body25_pose_detection(input_video, output_video)