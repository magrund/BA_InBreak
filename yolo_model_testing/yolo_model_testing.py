import cv2
from ultralytics import YOLO
import numpy as np
import os

COCO_KEYPOINT_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

yolo_color = (255, 0, 0)

model = YOLO("models/yolo11m-pose.pt")
confidence_threshold = 0.1

def yolo_pose_on_images(input_folder, output_folder):
    print("YOLO: Start processing images")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(input_folder):
        if not image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(input_folder, image_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"YOLO: Could not read image {image_name}")
            continue

        results = model(image)

        for result in results:
            kpts = result.keypoints.data[0].cpu().numpy()

            for kpt in kpts:
                x, y, confidence = kpt
                if confidence > confidence_threshold:
                    cv2.circle(image, (int(x), int(y)), 5, yolo_color, -1)

            for pair in COCO_KEYPOINT_PAIRS:
                if pair[0] < len(kpts) and pair[1] < len(kpts):
                    x1, y1, conf1 = kpts[pair[0]]
                    x2, y2, conf2 = kpts[pair[1]]
                    if conf1 > confidence_threshold and conf2 > confidence_threshold:
                        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), yolo_color, 2)

        output_path = os.path.join(output_folder, image_name)
        cv2.imwrite(output_path, image)
        print(f"YOLO: Processed and saved {image_name}")

    print("YOLO: All images processed.")

if __name__ == '__main__':
    input_folder = 'evaluation_dataset/images/'
    output_folder = 'output/yolo_pose'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    yolo_pose_on_images(input_folder, output_folder)