import csv
import os
import cv2

def check_frames_exist(video_id, start_frame, end_frame, frames_folder):
    for frame_number in range(start_frame, end_frame + 1):
        frame_file = os.path.join(frames_folder, f'{video_id}_img-{frame_number:06d}.jpg')
        if not os.path.exists(frame_file):
            return False
    return True

def export_specific_frame(video_name, frame_number, actual_frame_number, videos_folder, frames_folder):
    video_path = os.path.join(videos_folder, f'{video_name}.mp4')
    os.makedirs(frames_folder, exist_ok=True)

    video_id = video_name[:11]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Error opening video file {video_path}')
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, actual_frame_number)
    ret, frame = cap.read()
    if ret:
        frame_file = os.path.join(frames_folder, f'{video_id}_img-{frame_number:06d}.jpg')
        cv2.imwrite(frame_file, frame)
        print(f'Frame {frame_number} exported to {frame_file}')
    else:
        print(f'Error reading frame {actual_frame_number} from video {video_path}')

    cap.release()

def export_frames(video_id, start_frame, end_frame, videos_folder, frames_folder):
    for frame_number in range(start_frame, end_frame + 1):
        actual_frame_number = frame_number - start_frame
        video_name = f'{video_id}_{start_frame}-{end_frame}'
        export_specific_frame(video_name, frame_number, actual_frame_number, videos_folder, frames_folder)

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
                print(f'Frames for video {video_id} from frame {start_frame} to {end_frame} already exist')