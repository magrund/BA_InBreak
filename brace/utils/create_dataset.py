import os
import numpy as np
import shutil
import random
from PIL import Image
import re

def create_relative_bounding_box(keypoints, image_width, image_height, buffer):
    x_min = np.min(keypoints[:, 0])
    y_min = np.min(keypoints[:, 1])
    x_max = np.max(keypoints[:, 0])
    y_max = np.max(keypoints[:, 1])

    width = x_max - x_min
    height = y_max - y_min
    buffer_x = width * buffer
    buffer_y = height * buffer

    x_min = max(0, x_min - buffer_x)
    y_min = max(0, y_min - buffer_y)
    x_max = min(image_width, x_max + buffer_x)
    y_max = min(image_height, y_max + buffer_y)

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    x_center_rel = x_center / image_width
    y_center_rel = y_center / image_height
    width_rel = bbox_width / image_width
    height_rel = bbox_height / image_height

    return x_center_rel, y_center_rel, width_rel, height_rel

def get_annotation_content(npz_file_path, image_width, image_height):
    keypoints = np.load(npz_file_path)['coco_joints2d'][:, :2]
    
    bbox = create_relative_bounding_box(keypoints, image_width, image_height, buffer=0.8)
    
    content = f"0 {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
    
    for kp in keypoints:
        x_rel = kp[0] / image_width
        y_rel = kp[1] / image_height
        conf = 1  
        content += f" {x_rel} {y_rel} {conf}"
    
    content += "\n"
    return content

def split_data(destination_folder, split_ratio):
    train_image_dir = os.path.join(destination_folder, 'train/images')
    train_label_dir = os.path.join(destination_folder, 'train/labels')
    val_image_dir = os.path.join(destination_folder, 'val/images')
    val_label_dir = os.path.join(destination_folder, 'val/labels')
    
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(train_image_dir) if f.endswith('.jpg') or f.endswith('.png')]
    
    val_size = int(len(image_files) * split_ratio)
    random.seed(42)
    val_files = random.sample(image_files, val_size)
    
    for image_file in val_files:
        image_path = os.path.join(train_image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(train_label_dir, label_file)
        
        val_image_path = os.path.join(val_image_dir, image_file)
        val_label_path = os.path.join(val_label_dir, label_file)
        
        shutil.move(image_path, val_image_path)
        shutil.move(label_path, val_label_path)
        
        print(f"Moved: {image_file} & {label_file} to val.")

    print(f"Images - Train: {len(os.listdir(train_image_dir))}, Val: {len(os.listdir(val_image_dir))}")
    print(f"Labels - Train: {len(os.listdir(train_label_dir))}, Val: {len(os.listdir(val_label_dir))}")


def create_dataset_with_split(pairs, destination_folder, split_ratio):
    destination_folder_images = os.path.join(destination_folder, 'train/images')
    destination_folder_txt = os.path.join(destination_folder, 'train/labels')

    if not os.path.exists(destination_folder_images):
        os.makedirs(destination_folder_images)
    if not os.path.exists(destination_folder_txt):
        os.makedirs(destination_folder_txt)
    
    print("Creating dataset...")

    for npz_file, image_file in pairs:
        pair_number = pairs.index((npz_file, image_file)) + 1
        file_name_base = os.path.splitext(os.path.basename(npz_file))[0]


        base_name = os.path.basename(image_file)
        new_file_name = f"{file_name_base}.jpg"
        
        destination_image_path = os.path.join(destination_folder_images, new_file_name)
        
        if not os.path.exists(destination_image_path):
            with Image.open(image_file) as img:
                img = img.convert("RGB")
                img.save(destination_image_path, "JPEG")
        
        txt_file_name = f"{file_name_base}.txt"
        txt_file_path = os.path.join(destination_folder_txt, txt_file_name)
        
        if not os.path.exists(txt_file_path):
            with Image.open(image_file) as img:
                image_width, image_height = img.size
            content = get_annotation_content(npz_file, image_width, image_height)
            
            with open(txt_file_path, 'w') as f:
                f.write(content)

        print(f"Processed {pair_number}/{len(pairs)}: {new_file_name} & {txt_file_name}")

    print("Starting to split data...")
    split_data(destination_folder, split_ratio)
    print("Splting done & Dataset created.")
        