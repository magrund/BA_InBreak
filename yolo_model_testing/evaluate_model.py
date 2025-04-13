import cv2
import os
import math
from collections import defaultdict
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
        yolo_keypoints = get_yolo_keypoints(image, self.model)

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
        stats_lines.append(f"Detected images: {self.recognized_images} of {self.total_images}")
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

if __name__ == '__main__':
    image_folder = 'evaluation_dataset/images/'
    label_folder = 'evaluation_dataset/labels/'

    model_name = 'yolo11m-pose'

    model = YOLO('models/' + model_name + '.pt')
    output_folder = 'output/yolo_pose_' + model_name

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    evaluator = KeypointEvaluator(image_folder, label_folder, output_folder, model)
    evaluator.evaluate_dataset()
    evaluator.write_statistics("statistics.txt")
