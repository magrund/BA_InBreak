import csv
import os
import subprocess
from datetime import datetime, timedelta

def download_videos(csv_path, videos_folder):
    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_id = row['video_id']
            video_url = row['video_url']
            
            video_path = os.path.join(videos_folder, f'{video_id}.mp4')
            
            if not os.path.exists(video_path):
                print(f'Download {video_id} ...')
                subprocess.run([
                    'yt-dlp', 
                    '-f', 'bv*[vcodec^=avc1][height<=1080][ext=mp4]+ba[acodec^=mp4a]/mp4',
                    '-o', video_path, 
                    video_url
                ])
                print(f'Video {video_id} downloaded.')
            else:
                print(f'Video {video_id} already exists.')

def download_video_segments(videos_info_csv, segments_info_csv, segments_folder):
    videos_info = {}
    with open(videos_info_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_id = row['video_id']
            videos_info[row['url']] = {
                'video_id': video_id,
                'fps': float(row['fps'])
            }

    with open(segments_info_csv, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            video_url = row['url']
            segment_id = row['segment_id']
            start_time = row['start']
            end_time = row['end']

            if video_url not in videos_info:
                print(f'Did not find video info for {video_url} in {videos_info_csv}. Skipping segment {segment_id}.')
                continue

            video_id = videos_info[video_url]['video_id']
            fps = videos_info[video_url]['fps']

            start_time_parts = list(map(float, start_time.split(':')))
            end_time_parts = list(map(float, end_time.split(':')))
            start_seconds = start_time_parts[0] * 3600 + start_time_parts[1] * 60 + start_time_parts[2]
            end_seconds = end_time_parts[0] * 3600 + end_time_parts[1] * 60 + end_time_parts[2]
            start_frame = round(start_seconds * fps)
            end_frame = round(end_seconds * fps)

            end_time = (datetime.strptime(end_time, "%H:%M:%S.%f") + timedelta(seconds=1)).strftime("%H:%M:%S.%f")[:-3]


            segment_path = os.path.join(segments_folder, f'{video_id}_{start_frame}-{end_frame}.mp4')

            if not os.path.exists(segment_path):
                subprocess.run([
                    'yt-dlp',
                    '-f', 'bv*[vcodec^=avc1][height<=1080][ext=mp4]+ba[acodec^=mp4a]/mp4',
                    '--download-sections', f'*{start_time}-{end_time}',
                    '-o', segment_path,
                    video_url
                ])
                print(f'Downloaded segment {segment_id} from video {video_id}.')
            else:
                print(f'Segment {segment_id} from video {video_id} already exists.')