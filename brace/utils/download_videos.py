import csv
import os
import subprocess

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