from pose_mediapipe_with_tracking import mediapipe_with_tracking_pose_detection
from pose_mediapipe_no_tracking import mediapipe_no_tracking_pose_detection
from pose_movenet import movenet_pose_detection
from pose_yolo import yolo_pose_detection
from pose_openpose_coco import openpose_coco_pose_detection
from pose_openpose_body25 import openpose_body25_pose_detection
from pose_all_models import all_models_pose_detection

input_video = 'Handstand.MOV'
output_video = 'output'

def all_models(input_video, output_video):
    movenet_keypoints = movenet_pose_detection(input_video, output_video)
    yolo_keypoints = yolo_pose_detection(input_video, output_video)
    mediapipe_with_tracking_keypoints = mediapipe_with_tracking_pose_detection(input_video, output_video)
    mediapipe_no_tracking_keypoints = mediapipe_no_tracking_pose_detection(input_video, output_video)
    openpose_body25_keypoints = openpose_body25_pose_detection(input_video, output_video)
    openpose_coco_keypoints = openpose_coco_pose_detection(input_video, output_video)

    all_models_pose_detection(input_video, output_video, movenet_keypoints, yolo_keypoints, mediapipe_with_tracking_keypoints, mediapipe_no_tracking_keypoints, openpose_body25_keypoints, openpose_coco_keypoints)

if __name__ == '__main__':
    mediapipe_with_tracking_pose_detection(input_video, output_video)
    mediapipe_no_tracking_pose_detection(input_video, output_video)
    movenet_pose_detection(input_video, output_video)
    yolo_pose_detection(input_video, output_video)
    openpose_coco_pose_detection(input_video, output_video)
    openpose_body25_pose_detection(input_video, output_video)

    #all_models(input_video, output_video)