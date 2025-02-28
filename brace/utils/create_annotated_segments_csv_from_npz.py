import csv
import os

def create_dance_type_csvs(segments_csv, output_folder):
    segments = []
    with open(segments_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            segments.append(row)

    dance_types = ['powermove', 'toprock', 'footwork']
    dance_type_segments = {dance_type: [] for dance_type in dance_types}
    for segment in segments:
        dance_type = segment['dance_type']
        if dance_type in dance_type_segments:
            dance_type_segments[dance_type].append(segment)

    for dance_type, segments in dance_type_segments.items():
        output_csv = os.path.join(output_folder, f'{dance_type}_segments.csv')
        print(f'Creating {output_csv} ...')
        with open(output_csv, 'w', newline='') as csvfile:
            if segments:
                fieldnames = segments[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for segment in segments:
                    writer.writerow(segment)
            else:
                print(f'Found no segments for {dance_type}.')
        print(f'{output_csv} created.')

def check_npz_files_and_create_csv(segments_csv, npz_folder, output_folder):
    segments = []
    with open(segments_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            segments.append(row)

    annotated_segments = {'powermove': [], 'toprock': [], 'footwork': [], 'all': []}
    for segment in segments:
        video_id = segment['video_id']
        start_frame = int(segment['start_frame'])
        end_frame = int(segment['end_frame'])
        dance_type = segment['dance_type']
        
        for year_folder in os.listdir(npz_folder):
            year_path = os.path.join(npz_folder, year_folder)
            if os.path.isdir(year_path):
                for frame_number in range(start_frame, end_frame + 1):
                    npz_file = os.path.join(year_path, f'{video_id}/img-{frame_number:06d}.npz')
                    if os.path.exists(npz_file):
                        new_segment = segment.copy()
                        new_segment['start_frame'] = frame_number
                        new_segment['end_frame'] = frame_number
                        annotated_segments[dance_type].append(new_segment)
                        annotated_segments['all'].append(new_segment)

    for dance_type, segments in annotated_segments.items():
        output_csv = os.path.join(output_folder, f'annotated_{dance_type}_segments.csv')
        with open(output_csv, 'w', newline='') as csvfile:
            print(f'Creating {output_csv} ...')
            if segments:
                fieldnames = segments[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for segment in segments:
                    writer.writerow(segment)
            else:
                print(f'Found no segments for {dance_type}.')
            print(f'{output_csv} created.')

    for dance_type, segments in annotated_segments.items():
        total_frames = sum(int(segment['end_frame']) - int(segment['start_frame']) + 1 for segment in segments)
        print(f'Number of frames in {dance_type}: {total_frames}')