import os
import csv

def validate_data_pairs(segments_csv, npz_folder, frames_folder):
    is_dataset_complete = False
    npz_files = []
    frame_files = []
    missing_npz_files = []
    missing_frame_files = []
    pairs = []
    missing_pairs = []
    
    segments = []
    with open(segments_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            segments.append(row)
    
    for segment in segments:
        video_id = segment['video_id']
        start_frame = int(segment['start_frame'])
        end_frame = int(segment['end_frame'])
        
        for frame_number in range(start_frame, end_frame + 1):
            npz_file = os.path.join(npz_folder, f'{video_id}_img-{frame_number:06d}.npz')
            frame_file = os.path.join(frames_folder, f'{video_id}_img-{frame_number:06d}.jpg')
            npz_exists = os.path.exists(npz_file)
            frame_exists = os.path.exists(frame_file)
            if npz_exists and frame_exists:
                pairs.append((npz_file, frame_file))
            else:
                missing_pairs.append((npz_file, frame_file))
            if npz_exists:
                npz_files.append(npz_file)
            else:
                missing_npz_files.append(npz_file)
            if frame_exists:
                frame_files.append(frame_file)
            else:
                missing_frame_files.append(frame_file)

    if missing_pairs:
        for npz_file, frame_file in missing_pairs:
            print(f'Missing Pair: NPZ: {npz_file}, Frame: {frame_file}')
            
        print(f'Missing pairs: {len(missing_pairs)}')

    if missing_npz_files:
        print("The following .npz files are missing:")
        for file in missing_npz_files:
            print(file)

    if missing_frame_files:
        print("The following frames are missing:")
        for file in missing_frame_files:
            print(file)

    if missing_npz_files or missing_frame_files or missing_pairs:
        print(f'Pairs found: {len(pairs)} from {len(segments)}')
        print(f'Npz-files found: {len(npz_files)} from {len(segments)}')
        print(f'Frames found: {len(frame_files)} from {len(segments)}')
        print("Dataset is not complete.")

    if not missing_npz_files and not missing_frame_files and not missing_pairs:
        print("Dataset is complete.")
        is_dataset_complete = True
    
    return is_dataset_complete, pairs