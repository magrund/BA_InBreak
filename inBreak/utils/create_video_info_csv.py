import yt_dlp
import csv

def get_video_info(urls, output_file):
    results = []
    
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            try:
                info = ydl.extract_info(url, download=False)
                
                channel = info.get('channel', 'Unknown')
                channel_id = info.get('channel_id', 'Unknown')
                duration = info.get('duration_string', 'Unknown')
                video_id = info.get('id', 'Unknown')
                video_title = info.get('title', 'Unknown')
                
                highest_resolution = None
                highest_fps = None
                formats = info.get('formats', [])
                
                for f in formats:
                    if f.get('vcodec') != 'none':
                        width = f.get('width', 0)
                        height = f.get('height', 0)
                        fps = f.get('fps', None)
                        if width > 0 and height > 0:
                            resolution = f"{width}x{height}"
                            if highest_resolution is None or width * height > int(highest_resolution.split('x')[0]) * int(highest_resolution.split('x')[1]):
                                highest_resolution = resolution
                                highest_fps = fps
                
                results.append({
                    'video_id': video_id,
                    'duration': duration,
                    'fps': highest_fps,
                    'resolution': highest_resolution,
                    'channel': channel,
                    'channel_id': channel_id,
                    'url': url,
                    'title': video_title,
                })
            except Exception as e:
                print(f"Error processing {url}: {e}")
                results.append({
                    'video_id': 'Unknown',
                    'duration': None,
                    'fps': None,
                    'resolution': None,
                    'channel': 'Unknown',
                    'channel_id': 'Unknown',
                    'url': url,
                    'title': 'Unknown',
                })
    
    with open(output_file, mode='w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['video_id', 'duration', 'fps', 'resolution', 'channel', 'channel_id', 'url', 'title']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(results)

def load_urls_from_csv(input_url_csv):
    urls = []
    with open(input_url_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            urls.append(row["url"])
    return urls

def create_video_info_csv(url_file, output_file):
    print(f"Loading URLs from " + url_file)
    urls = load_urls_from_csv(url_file)
    print("URLs loaded successfully.")
    if urls:
        print("Getting video information...")
        get_video_info(urls, output_file)
        print(f"Video information saved successfully in " + output_file)