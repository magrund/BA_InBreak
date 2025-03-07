import os
import csv

def check_if_all_frames_exist_from_csv(csv_path, frames_folder):
    segments = []

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            segments.append(row)

    total_frames_in_csv = 0
    for segment in segments:
        start_frame = int(segment['start_frame'])
        end_frame = int(segment['end_frame'])
        total_frames_in_csv += (end_frame - start_frame + 1)

    missing_frames = []
    total_frames_in_folders = 0
    for segment in segments:
        video_id = segment['video_id']
        start_frame = int(segment['start_frame'])
        end_frame = int(segment['end_frame'])
        
        for frame_number in range(start_frame, end_frame + 1):
            frame_file = os.path.join(frames_folder, f'{video_id}_img-{frame_number:06d}.jpg')
            if os.path.exists(frame_file):
                total_frames_in_folders += 1
            else:
                missing_frames.append(frame_file)

    if total_frames_in_folders != total_frames_in_csv:
        if total_frames_in_folders == 0:
            print("No frames found. Error in exporting frames.")
        else:
            print(f"There are missing frames. Number of Frames missing: " + str(total_frames_in_csv - total_frames_in_folders))
            for missing_frame in missing_frames:
                print(f"Missing frame: " + missing_frame)
    else:
        print(f"All frames exported successfully. Number of Frames: " + str(total_frames_in_folders))