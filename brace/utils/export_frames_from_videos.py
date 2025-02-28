import csv
import os
import cv2

def export_frames(video_id, start_frame, end_frame, videos_folder, frames_folder):
    video_file_path = os.path.join(videos_folder, f'{video_id}.mp4')
    os.makedirs(frames_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_file_path)
    if not cap.isOpened():
        print(f'Error opening video file {video_file_path}')
        return

    for frame_number in range(start_frame, end_frame + 1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
        ret, frame = cap.read()
        if ret:
            frame_file_path = os.path.join(frames_folder, f'{video_id}_img-{frame_number:06d}.jpg')
            cv2.imwrite(frame_file_path, frame)
            print(f'Frame {frame_number} exported to {frame_file_path}')
        else:
            print(f'Error reading frame {frame_number} from video {video_file_path}')

    cap.release()

def check_frames_exist(video_id, start_frame, end_frame, frames_folder):
    for frame_number in range(start_frame, end_frame + 1):
        frame_file_path = os.path.join(frames_folder, f'{video_id}_img-{frame_number:06d}.jpg')
        if not os.path.exists(frame_file_path):
            return False
    return True

def export_frames_from_segments(segments_csv, videos_folder, frames_folder):
    with open(segments_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_id = row['video_id']
            start_frame = int(row['start_frame'])
            end_frame = int(row['end_frame'])
            if not check_frames_exist(video_id, start_frame, end_frame, frames_folder):
                export_frames(video_id, start_frame, end_frame, videos_folder, frames_folder)
            else:
                print(f'Frames {start_frame} to {end_frame} for video {video_id} already exist.')