import cv2
import os
import math
from collections import defaultdict
from ultralytics import YOLO
import csv
import re

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
confidence_threshold = 0.5

KEYPOINT_NAMES = {
    0: "Nose",
    1: "Left Eye",
    2: "Right Eye",
    3: "Left Ear",
    4: "Right Ear",
    5: "Left Shoulder",
    6: "Right Shoulder",
    7: "Left Elbow",
    8: "Right Elbow",
    9: "Left Wrist",
    10: "Right Wrist",
    11: "Left Hip",
    12: "Right Hip",
    13: "Left Knee",
    14: "Right Knee",
    15: "Left Ankle",
    16: "Right Ankle"
}

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
        speed = result.speed
        total_time = speed.get('preprocess', 0.0) + speed.get('inference', 0.0) + speed.get('postprocess', 0.0)
        return [(x, y, conf) for x, y, conf in kpts], total_time
    return [], 0.0


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

class KeypointEvaluator:
    def __init__(self, image_folder, label_folder, output_folder, model):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.output_folder = output_folder
        self.model = model

        self.overall_errors = []
        self.overall_raw_errors = []
        self.overall_confidences = []
        self.stats_by_type = defaultdict(list)
        self.stats_by_type_raw = defaultdict(list)
        self.stats_by_gender = defaultdict(list)
        self.stats_by_gender_raw = defaultdict(list)
        self.stats_by_bodypart = defaultdict(list)
        self.stats_by_bodypart_raw = defaultdict(list)
        self.stats_per_image = {}
        self.recognized_images = 0  
        self.total_times = []
        self.face_indices = set(range(5))
        self.face_weight = 0.5
        self.error_clip = 50.0
        self.total_images = len([f for f in os.listdir(self.image_folder) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    def compute_euclidean_distance(self, p1, p2):
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    def evaluate_image(self, image_name):
        image_path = os.path.join(self.image_folder, image_name)
        label_path = os.path.join(self.label_folder, os.path.splitext(image_name)[0] + '.txt')
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error Loading: {image_path}")
            return

        h, w = image.shape[:2]
        label_keypoints = get_label_keypoints(label_path, w, h)
        yolo_keypoints, total_time = get_yolo_keypoints(image, self.model)
        self.total_times.append(total_time)

        image_vis = image.copy()
        image_vis = draw_keypoints(image_vis, label_keypoints, LABEL_COLOR, is_yolo=False)
        image_vis = draw_keypoints(image_vis, yolo_keypoints, YOLO_COLOR, is_yolo=True)

        output_path = os.path.join(self.output_folder, "images")
        os.makedirs(output_path, exist_ok=True)
        cv2.imwrite(os.path.join(output_path, image_name), image_vis)

        errors = []
        raw_errors = []
        confidences = []
        for idx, label_kp in enumerate(label_keypoints):
            weight = self.face_weight if idx in self.face_indices else 1.0
        
            if idx >= len(yolo_keypoints):
                clipped_error = self.error_clip
                raw_error = self.error_clip
                errors.append(clipped_error * weight)
                raw_errors.append(raw_error * weight)
                confidences.append(0)
                self.stats_by_bodypart[idx].append(clipped_error)
                self.stats_by_bodypart_raw[idx].append(raw_error)
                self.overall_errors.append(clipped_error * weight)
                self.overall_raw_errors.append(raw_error * weight)
                continue

            yolo_kp = yolo_keypoints[idx]
            conf = yolo_kp[2]
            if conf < 0.5:
                clipped_error = self.error_clip
                raw_error = self.error_clip
                errors.append(clipped_error * weight)
                raw_errors.append(raw_error * weight)
                confidences.append(0)
                self.stats_by_bodypart[idx].append(clipped_error)
                self.stats_by_bodypart_raw[idx].append(raw_error)
                self.overall_errors.append(clipped_error * weight)
                self.overall_raw_errors.append(raw_error * weight)
            else:
                raw_error = self.compute_euclidean_distance(label_kp, yolo_kp[:2])
                clipped_error = min(raw_error, self.error_clip)
                errors.append(clipped_error * weight)
                raw_errors.append(raw_error * weight)
                confidences.append(conf)
                self.stats_by_bodypart[idx].append(clipped_error)
                self.stats_by_bodypart_raw[idx].append(raw_error)
                self.overall_errors.append(clipped_error * weight)
                self.overall_raw_errors.append(raw_error * weight)
        
        self.overall_confidences.extend(confidences)

        mean_error = sum(errors) / len(errors) if errors else 0
        mean_raw = sum(raw_errors) / len(raw_errors) if raw_errors else 0
        mean_conf = sum(confidences) / len(confidences) if confidences else 0
        self.stats_per_image[image_name] = (mean_error, mean_raw, mean_conf)

        if mean_conf > 0:
            self.recognized_images += 1

        basename = os.path.splitext(image_name)[0]
        parts = basename.split('_')
        if len(parts) >= 3:
            gender = parts[1].upper()
            typ = parts[2].capitalize()
            self.stats_by_gender[gender].append(mean_error)
            self.stats_by_gender_raw[gender].append(mean_raw)
            self.stats_by_type[typ].append(mean_error)
            self.stats_by_type_raw[typ].append(mean_raw)

    def evaluate_dataset(self):
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        for image_name in os.listdir(self.image_folder):
            if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                self.evaluate_image(image_name)

    def generate_statistics(self):
        stats_lines = []
        avg_total_time = sum(self.total_times) / len(self.total_times) if self.total_times else 0
        sum_total_time = sum(self.total_times)

        stats_lines.append(f"Detected images: {self.recognized_images} of {self.total_images}")
        stats_lines.append(f"Average Total Inference Time:        {avg_total_time:.2f} ms")
        stats_lines.append(f"Total Time for All Images:           {sum_total_time/1000:.2f} seconds")
        stats_lines.append("")

        overall_avg_error = sum(self.overall_errors) / len(self.overall_errors) if self.overall_errors else 0
        overall_avg_raw = sum(self.overall_raw_errors) / len(self.overall_raw_errors) if self.overall_raw_errors else 0
        overall_avg_conf = sum(self.overall_confidences) / len(self.overall_confidences) if self.overall_confidences else 0
        stats_lines.append("Overall Statistics:")
        stats_lines.append(f"  Mean Deviation:                      {overall_avg_error:.2f} px ({overall_avg_raw:.2f} px)")
        stats_lines.append(f"  Mean Confidence Score:              {overall_avg_conf:.2f}")
        stats_lines.append("")

        stats_lines.append("Statistics by Movement Type:")
        for typ in sorted(self.stats_by_type.keys()):
            errors_clipped = self.stats_by_type[typ]
            errors_raw = self.stats_by_type_raw[typ]
            avg = sum(errors_clipped) / len(errors_clipped) if errors_clipped else 0
            raw_avg = sum(errors_raw) / len(errors_raw) if errors_raw else 0
            stats_lines.append(f"  {typ:<10} Mean Deviation:            {avg:.2f} px ({raw_avg:.2f} px)")
        stats_lines.append("")

        stats_lines.append("Statistics by Gender:")
        for gender in sorted(self.stats_by_gender.keys()):
            errors_clipped = self.stats_by_gender[gender]
            errors_raw = self.stats_by_gender_raw[gender]
            avg = sum(errors_clipped) / len(errors_clipped) if errors_clipped else 0
            raw_avg = sum(errors_raw) / len(errors_raw) if errors_raw else 0
            stats_lines.append(f"  {gender:<6} Mean Deviation:           {avg:.2f} px ({raw_avg:.2f} px)")
        stats_lines.append("")

        stats_lines.append("Statistics by Body Part:")
        for part_idx in sorted(self.stats_by_bodypart.keys()):
            errors_clipped = self.stats_by_bodypart[part_idx]
            errors_raw = self.stats_by_bodypart_raw[part_idx]
            avg = sum(errors_clipped) / len(errors_clipped) if errors_clipped else 0
            raw_avg = sum(errors_raw) / len(errors_raw) if errors_raw else 0
            kp_name = KEYPOINT_NAMES.get(part_idx, f"Keypoint {part_idx}")
            stats_lines.append(f"  {part_idx:>2} ({kp_name:<12}) Mean Deviation:   {avg:.2f} px ({raw_avg:.2f} px)")
        stats_lines.append("")

        stats_lines.append("Statistics per Image:")
        def sort_key(name):
            base = os.path.splitext(name)[0]
            parts = base.split('_')
            try:
                return int(parts[0])
            except:
                return 1e9
        for image_name in sorted(self.stats_per_image.keys(), key=sort_key):
            mean_error, mean_raw, mean_conf = self.stats_per_image[image_name]
            stats_lines.append(f"  {image_name:<25} Deviation: {mean_error:.2f} px ({mean_raw:.2f} px) | Confidence: {mean_conf:.2f}")
        stats_lines.append("")
        return "\n".join(stats_lines)


    def write_statistics(self, filename="statistics.txt"):
        stats_text = self.generate_statistics()
        output_path = os.path.join(self.output_folder, filename)
        with open(output_path, 'w') as f:
            f.write(stats_text)
        print(f"Statistic saved in '{output_path}'.")

def extract_value(value_str):
    match = re.search(r"\((\d+(\.\d+)?)\s*px\)", value_str)
    if match:
        return float(match.group(1))
    return None

def remove_px_and_parentheses(value_str):
    value_without_px = re.sub(r" px", "", value_str)
    value_without_parentheses = re.sub(r"\(.*\)", "", value_without_px)
    return value_without_parentheses.strip()

def generate_model_summary_table(output_base, output_csv="model_summary.csv"):
    all_rows = []
    headers = [
        "Model", "Detected", "Overall Deviation", "Raw Overall Deviation", "Mean Confidence", 
        "Freeze", "Powermove", "Transition", "Bboy", "Bgirl", 
        "Raw Freeze", "Raw Powermove", "Raw Transition", "Raw Bboy", "Raw Bgirl",
        "Average Inference Time (ms)", "Total Time (s)"
    ]

    movement_keys = ["Freeze", "Powermove", "Transition"]
    gender_keys = ["Bboy", "Bgirl"]

    movement_aliases = {
        "freeze": "Freeze",
        "power": "Powermove",
        "powermove": "Powermove",
        "transition": "Transition"
    }

    gender_aliases = {
        "bboy": "Bboy",
        "bgirl": "Bgirl"
    }

    for folder_name in os.listdir(output_base):
        folder_path = os.path.join(output_base, folder_name)
        stats_file = os.path.join(folder_path, "statistics.txt")

        if not os.path.isfile(stats_file):
            continue

        with open(stats_file, 'r') as f:
            lines = f.readlines()

        model_name = folder_name.replace('_yolo-pose', '')
        row = {key: "" for key in headers}
        row["Model"] = model_name

        for i, line in enumerate(lines):
            line_clean = line.strip()
            lower_line = line_clean.lower()

            if "detected images:" in lower_line:
                parts = line_clean.split(":")[1].split("of")
                row["Detected"] = parts[0].strip()

            elif "average total inference time" in lower_line:
                match = re.search(r"([\d.]+)\s*ms", line_clean)
                if match:
                    row["Average Inference Time (ms)"] = match.group(1)

            elif "total time for all images" in lower_line:
                match = re.search(r"([\d.]+)\s*seconds", line_clean)
                if match:
                    row["Total Time (s)"] = match.group(1)

            elif "mean deviation:" in lower_line and "overall" in lines[i - 1].lower():
                row["Overall Deviation"] = remove_px_and_parentheses(line_clean.split(":")[1].strip())
                row["Raw Overall Deviation"] = extract_value(line_clean)

            elif "mean confidence score:" in lower_line:
                row["Mean Confidence"] = remove_px_and_parentheses(line_clean.split(":")[1].strip())

            else:
                for key, std_name in movement_aliases.items():
                    if key in lower_line:
                        parts = line_clean.split("Mean Deviation:")
                        if len(parts) == 2:
                            value = parts[1].strip()
                            row[std_name] = remove_px_and_parentheses(value)
                            row[f"Raw {std_name}"] = extract_value(value)
                for key, std_name in gender_aliases.items():
                    if key in lower_line:
                        parts = line_clean.split("Mean Deviation:")
                        if len(parts) == 2:
                            value = parts[1].strip()
                            row[std_name] = remove_px_and_parentheses(value)
                            row[f"Raw {std_name}"] = extract_value(value)

        all_rows.append(row)

    out_path = os.path.join(output_base, output_csv)
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(all_rows)


if __name__ == '__main__':
    image_folder = 'DATASET_PATH/images/'
    label_folder = 'DATASET_PATH/labels/'
    model_dir = 'MODELS_PATH/'
    output_base = 'OUTPUT_PATH/'

    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]

    for model_file in model_files:
        model_name = os.path.splitext(model_file)[0]
        output_folder = os.path.join(output_base, model_name)
        stats_file = os.path.join(output_folder, "statistics.txt")

        if os.path.exists(stats_file):
            print(f"Skipping {model_name}: Statistics already exist.")
            continue

        print(f"Evaluating: {model_name}")
        model_path = os.path.join(model_dir, model_file)
        model = YOLO(model_path)

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        evaluator = KeypointEvaluator(image_folder, label_folder, output_folder, model)
        evaluator.evaluate_dataset()
        evaluator.write_statistics("statistics.txt")

        print(f"Done: {model_name}")

    generate_model_summary_table(output_base, "model_summary.csv")