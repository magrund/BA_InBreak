from pose_mediapipe import mediapipe_pose_detection
from pose_movenet import movenet_pose_detection
from pose_yolo import yolo_pose_detection
from pose_openpose import openpose_pose_detection

input_video = 'Handstand.MOV'
output_video = 'output'

if __name__ == '__main__':
    mediapipe_pose_detection(input_video, output_video)
    movenet_pose_detection(input_video, output_video)
    yolo_pose_detection(input_video, output_video)
    openpose_pose_detection(input_video, output_video)