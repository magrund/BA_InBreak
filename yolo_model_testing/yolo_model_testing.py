import cv2
import os
from ultralytics import YOLO

COCO_KEYPOINT_PAIRS = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 6), (5, 11), (6, 12),
    (11, 12), (11, 13), (13, 15),
    (12, 14), (14, 16)
]

LABEL_COLOR = (0, 255, 0)
YOLO_COLOR  = (255, 255, 0)
confidence_threshold = 0.1

def get_label_keypoints(label_path, width, height):
    with open(label_path, 'r') as f:
        data = f.readline().strip().split()
        keypoints = [(float(data[i]) * width, float(data[i+1]) * height) 
                     for i in range(5, len(data), 3)]
    return keypoints

def get_yolo_keypoints(image, model):
    results = model(image)
    for result in results:
        kpts = result.keypoints.data[0].cpu().numpy()
        return [(x, y, conf) for x, y, conf in kpts]
    return []

def draw_keypoints(image, keypoints, color, is_yolo=False):
    if not is_yolo:
        alpha = 0.6
        overlay = image.copy()
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(overlay, (x, y), 5, color, -1)
        for pair in COCO_KEYPOINT_PAIRS:
            if pair[0] < len(keypoints) and pair[1] < len(keypoints):
                pt1 = (int(keypoints[pair[0]][0]), int(keypoints[pair[0]][1]))
                pt2 = (int(keypoints[pair[1]][0]), int(keypoints[pair[1]][1]))
                cv2.line(overlay, pt1, pt2, color, 2)
        image = cv2.addWeighted(image, alpha, overlay, 1 - alpha, 0)
    else:
        for kp in keypoints:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(image, (x, y), 5, color, -1)
        for pair in COCO_KEYPOINT_PAIRS:
            if pair[0] < len(keypoints) and pair[1] < len(keypoints):
                x1, y1, conf1 = keypoints[pair[0]]
                x2, y2, conf2 = keypoints[pair[1]]
                if conf1 > confidence_threshold and conf2 > confidence_threshold:
                    cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
    return image

def yolo_model_testing(image_folder, label_folder, output_folder, model):
    for image_name in os.listdir(image_folder):
        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_folder, image_name)
            image = cv2.imread(image_path)
            h, w = image.shape[:2]
            label_path = os.path.join(label_folder, os.path.splitext(image_name)[0] + '.txt')
            label_keypoints = get_label_keypoints(label_path, w, h)
            yolo_keypoints = get_yolo_keypoints(image, model)
            image = draw_keypoints(image, label_keypoints, LABEL_COLOR, is_yolo=False)
            image = draw_keypoints(image, yolo_keypoints, YOLO_COLOR, is_yolo=True)
            cv2.imwrite(os.path.join(output_folder, image_name), image)

if __name__ == '__main__':
    image_folder = 'evaluation_dataset/images/'
    label_folder = 'evaluation_dataset/labels/'
    output_folder = 'output/yolo_pose_simple2'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    model = YOLO("models/best_mix.pt")
    yolo_model_testing(image_folder, label_folder, output_folder, model)
