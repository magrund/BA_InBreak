import cv2
import numpy as np

# Modell-Dateien
proto_file = "models/coco/pose_deploy_linevec.prototxt"
weights_file = "models/coco/pose_iter_440000.caffemodel"

# Anzahl der Keypoints im COCO-Modell
n_points = 18

# Definiert die Keypoint-Verbindungen (Skelett)
pose_pairs = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8),
              (8, 9), (9, 10), (1, 11), (11, 12), (12, 13), (1, 0),
              (0, 14), (14, 16), (0, 15), (15, 17)]

# Netzwerk laden
net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)
print("OpenPose: Netzwerk geladen")

def openpose_pose_detection(input_video, output_video):
    print("Pose Estimation: Starte Videoverarbeitung")

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print("Fehler: Video konnte nicht geöffnet werden!")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_video + "_pose.mp4", fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        print("Frame: ", frame.shape)
        frame_height, frame_width = frame.shape[:2]

        # Bild in das richtige Format für das Netz umwandeln
        inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (368, 368), (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inp_blob)
        output = net.forward()
        print("Output: ", output.shape)

        points = []

        for i in range(n_points):
            heatmap = output[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatmap)

            x = int(frame_width * point[0] / output.shape[3])
            y = int(frame_height * point[1] / output.shape[2])
            print("Point: ", x, y, conf)

            if conf > 0.1:  # Minimum Confidence
                points.append((x, y))
                cv2.circle(frame, (x, y), 5, (0, 255, 0), thickness=-1, lineType=cv2.FILLED)
            else:
                points.append(None)

        # Verbinde die Keypoints mit Linien (Skelett)
        for pair in pose_pairs:
            part_a, part_b = pair
            if points[part_a] and points[part_b]:
                cv2.line(frame, points[part_a], points[part_b], (0, 0, 255), 2, lineType=cv2.LINE_AA)

        print("Done Frame: ", frame.shape)
        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Pose Estimation abgeschlossen:", output_video + "_pose.mp4")