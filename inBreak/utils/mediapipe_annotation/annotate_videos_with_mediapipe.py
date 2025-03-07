import cv2
import mediapipe as mp
import os
import re
import xml.etree.ElementTree as ET
from utils.mediapipe_annotation.generateAnnotationXML import generate_xml

COCO_KEYPOINTS = [
    ("nose", 0),
    ("left_eye", 2),
    ("right_eye", 5),
    ("left_ear", 7),
    ("right_ear", 8),
    ("left_shoulder", 11),
    ("right_shoulder", 12),
    ("left_elbow", 13),
    ("right_elbow", 14),
    ("left_wrist", 15),
    ("right_wrist", 16),
    ("left_hip", 23),
    ("right_hip", 24),
    ("left_knee", 25),
    ("right_knee", 26),
    ("left_ankle", 27),
    ("right_ankle", 28)
]

template_path = "inBreak/utils/mediapipe_annotation/annotation_template.xml"

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def extract_start_frame(video_file):
    parts = video_file.split('_')
    if len(parts) > 1:
        frame_range = parts[-1].split('-')
        if len(frame_range) > 1:
            return int(frame_range[0])
    return 0

def extract_keypoints(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"OpenCV: Couldn't read video stream from file {video_path}")
        return []

    keypoints_list = []
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_number = extract_start_frame(video_path)
    previous_keypoints = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        keypoints = {"frame_number": frame_number}
        if results.pose_landmarks:
            for name, index in COCO_KEYPOINTS:
                landmark = results.pose_landmarks.landmark[index]
                if landmark.visibility < 0.1:
                    if previous_keypoints and name in previous_keypoints:
                        keypoints[name] = previous_keypoints[name]
                    else:
                        keypoints[name] = {"x": 10, "y": 10}
                else:
                    keypoints[name] = {
                        "x": landmark.x * width,
                        "y": landmark.y * height
                    }
            previous_keypoints = keypoints
        else:
            if previous_keypoints:
                for name, _ in COCO_KEYPOINTS:
                    keypoints[name] = previous_keypoints.get(name, {"x": 10, "y": 10})
            else:
                for name, _ in COCO_KEYPOINTS:
                    keypoints[name] = {"x": 10, "y": 10}
        keypoints_list.append(keypoints)
        frame_number += 1

    cap.release()
    return keypoints_list

def annotate_segments_in_folder(folder_path, output_folder):
    video_data = {}

    for file in os.listdir(folder_path):
        if file.endswith(".mp4"):
            video_path = os.path.join(folder_path, file)
            video_file = os.path.splitext(file)[0]
            video_id = video_file[:11]
            start_frame = extract_start_frame(video_file)
            keypoints_list = extract_keypoints(video_path)

            if video_id not in video_data:
                video_data[video_id] = []
            video_data[video_id].append((start_frame, keypoints_list))
            print(f"Extracted {len(keypoints_list)} keypoints from {video_file}")
            print(f"Total keypoints for {video_id}: {len(video_data[video_id])}")

    for video_id, keypoints_data in video_data.items():
        keypoints_data.sort(key=lambda x: x[0])
        all_keypoints = []
        for start_frame, keypoints_list in keypoints_data:
            all_keypoints.extend(keypoints_list)
        output_path = os.path.join(output_folder, f"{video_id}.xml")
        print(f"Generating XML for {len(all_keypoints)} frames for video {video_id}")
        generate_xml(all_keypoints, template_path, output_path, video_id)