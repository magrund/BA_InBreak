import os
import cv2
import random

keypoint_labels = [
    "Nose", "Left eye", "Right eye", "Left ear", "Right ear",
    "Left shoulder", "Right shoulder", "Left elbow", "Right elbow",
    "Left wrist", "Right wrist", "Left hip", "Right hip",
    "Left knee", "Right knee", "Left ankle", "Right ankle"
]

def draw_label_on_image(image_path, label_path, output_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Das Bild konnte nicht geladen werden: {image_path}")
        return
    
    image_height, image_width, _ = image.shape
    
    with open(label_path, 'r') as f:
        data = f.readline().strip().split()
    
    bbox = [float(data[1]), float(data[2]), float(data[3]), float(data[4])]
    keypoints = [(float(data[i]), float(data[i+1]), int(data[i+2])) for i in range(5, len(data), 3)]
    
    x_center = bbox[0] * image_width
    y_center = bbox[1] * image_height
    width = bbox[2] * image_width
    height = bbox[3] * image_height
    
    x_min = int(x_center - width / 2)
    y_min = int(y_center - height / 2)
    x_max = int(x_center + width / 2)
    y_max = int(y_center + height / 2)
    
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
    
    for idx, (x, y, v) in enumerate(keypoints):
        x = int(x * image_width)
        y = int(y * image_height)
        if v > 0:
            cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(image, str(idx + 1), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    left_space = x_min
    right_space = image_width - x_max
    legend_x = x_max + 10 if right_space > left_space else x_min - 200
    
    for idx, label in enumerate(keypoint_labels):
        cv2.putText(image, f"{idx + 1}: {label}", (legend_x, y_min + 20 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imwrite(output_path, image)
    
def process_images(dataset_folder, output_folder, num_images):
    image_folder = os.path.join(dataset_folder, 'train/images')
    label_folder = os.path.join(dataset_folder, 'train/labels')

    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]
    random.seed(42)
    random.shuffle(image_files)
    image_files = image_files[:num_images]
    
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        label_file = image_file.replace('.jpg', '.txt')
        label_path = os.path.join(label_folder, label_file)
        output_path = os.path.join(output_folder, image_file)

        if not os.path.exists(output_path):
            os.makedirs(output_folder, exist_ok=True)
        
        draw_label_on_image(image_path, label_path, output_path)
        print(f"Image saved: {output_path}")