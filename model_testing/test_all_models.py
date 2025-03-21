from pose_mediapipe import mediapipe_pose_detection
from pose_movenet import movenet_pose_detection
from pose_yolo import yolo_pose_detection
from pose_openpose_coco import openpose_coco_pose_detection
from pose_openpose_body25 import openpose_body25_pose_detection

input_video = 'media.mp4'
output_video = 'output'

if __name__ == '__main__':
    #mediapipe_pose_detection(input_video, output_video)
    #movenet_pose_detection(input_video, output_video)
    #yolo_pose_detection(input_video, output_video)
    #openpose_coco_pose_detection(input_video, output_video)
    openpose_body25_pose_detection(input_video, output_video)