import os
import shutil
import random
import numpy as np
from PIL import Image

def reformat_dataset(dataset_folder):
    images_folder = os.path.join(dataset_folder, 'images/train')
    labels_folder = os.path.join(dataset_folder, 'labels/train')

    train_image_dir = os.path.join(dataset_folder, 'train/images')
    train_label_dir = os.path.join(dataset_folder, 'train/labels')
    val_image_dir = os.path.join(dataset_folder, 'val/images')
    val_label_dir = os.path.join(dataset_folder, 'val/labels')

    os.makedirs(train_image_dir, exist_ok=True)
    os.makedirs(train_label_dir, exist_ok=True)
    os.makedirs(val_image_dir, exist_ok=True)
    os.makedirs(val_label_dir, exist_ok=True)

    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg') or f.endswith('.png')]
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]

    print("Start restructuring dataset")

    for image_file in image_files:
        shutil.move(os.path.join(images_folder, image_file), os.path.join(train_image_dir, image_file))
        print(f"Moved: {image_file} to train.")

    os.rmdir(images_folder)
    os.rmdir(dataset_folder + '/images')

    for label_file in label_files:
        shutil.move(os.path.join(labels_folder, label_file), os.path.join(train_label_dir, label_file))
        print(f"Moved: {label_file} to train.")

    os.rmdir(labels_folder)
    os.rmdir(dataset_folder + '/labels')

    print("Done restructuring dataset")

def create_relative_bounding_box_and_update_label(label_file, image_width, image_height, buffer):
    with open(label_file, 'r') as file:
        lines = file.readlines()

    x_coords = []
    y_coords = []
    updated_lines = []
    for line in lines:
        parts = line.strip().split()
        keypoints = parts[5:]

        for i in range(0, len(keypoints), 3):
                x = float(keypoints[i])
                y = float(keypoints[i + 1])

                x_coords.append(x)
                y_coords.append(y)
                
        x_min = min(x_coords)* image_width
        x_max = max(x_coords)* image_width
        y_min = min(y_coords)* image_height
        y_max = max(y_coords)* image_height

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

        updated_line = f"{0} {x_center_rel} {y_center_rel} {width_rel} {height_rel}"

        if len(parts) > 5:
            updated_line += " " + " ".join(parts[5:])

        updated_line += "\n"
        updated_lines.append(updated_line)
    else:
        updated_lines.append(line)

    with open(label_file, 'w') as file:
        file.writelines(updated_lines)

    print(f"Die neue Bounding Box wurde in der Datei '{label_file}' gespeichert.")

def reformat_labels(dataset_folder):
    labels_folder = os.path.join(dataset_folder, 'train/labels')
    label_files = [f for f in os.listdir(labels_folder) if f.endswith('.txt')]

    print("Start reformatting labels")

    for label_file in label_files:
        file_path = os.path.join(labels_folder, label_file)

        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = [line.replace(" 2", " 1") for line in lines]

        with open(file_path, 'w') as f:
            f.writelines(new_lines)

        print(f"Reformatted: {label_file}")

    print("Done reformatting labels")

def split_data_and_add_bounding_box(dataset_folder, split_ratio, buffer):
    reformat_dataset(dataset_folder)
    reformat_labels(dataset_folder)

    train_image_dir = os.path.join(dataset_folder, 'train/images')
    train_label_dir = os.path.join(dataset_folder, 'train/labels')
    val_image_dir = os.path.join(dataset_folder, 'val/images')
    val_label_dir = os.path.join(dataset_folder, 'val/labels')

    print("Start calculating new bounding boxes")
    
    image_files = [f for f in os.listdir(train_image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    for image_file in image_files:
        image_path = os.path.join(train_image_dir, image_file)
        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(train_label_dir, label_file)

        print(f"Processing: {label_file} -> {label_path}")

        with Image.open(image_path) as img:
            image_width, image_height = img.size

        create_relative_bounding_box_and_update_label(label_path, image_width, image_height, buffer)

    
    print("Done calculating new bounding boxes")
    
    print("Start splitting dataset")

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

    print("Done splitting dataset")