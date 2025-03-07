import csv

def time_to_frames(time_str, fps):
    time_parts = time_str.split(":")
    if len(time_parts) == 2:
        minutes, seconds = time_parts
        hours = 0
    elif len(time_parts) == 3:
        hours, minutes, seconds = time_parts
    else:
        raise ValueError("Invalid time format")
    
    seconds_parts = seconds.split(".")
    if len(seconds_parts) == 2:
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1])
        milliseconds = milliseconds / 1000
    else:
        seconds = int(seconds_parts[0])
        milliseconds = 0

    total_seconds = int(hours) * 3600 + int(minutes) * 60 + seconds + milliseconds
    return round(total_seconds * fps)

def create_video_segments_csv(videos_info_file, segments_info_file, output_file):
    video_info = {}

    print(f"Reading video info from " + videos_info_file)

    with open(videos_info_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_info[row["url"]] = {
                "video_id": row["video_id"],
                "fps": float(row["fps"]),
                "channel_id": row["channel_id"]
            }

    print("Video info loaded")
    
    results = []
    
    print(f"Reading segments info from " + segments_info_file)

    with open(segments_info_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        current_video_id = None
        segment_id = 0
        
        for row in reader:
            url = row["url"]
            if url in video_info:
                video_data = video_info[url]
                fps = video_data["fps"]
                video_id = video_data["video_id"]
                channel_id = video_data["channel_id"]
                
                if current_video_id != video_id:
                    segment_id = 0
                    current_video_id = video_id
                
                start_frame = time_to_frames(row["start"], fps)
                end_frame = time_to_frames(row["end"], fps)
                
                results.append({
                    "video_id": video_id,
                    "segment_id": segment_id,
                    "start_frame": start_frame,
                    "end_frame": end_frame,
                    "type": row["type"],
                    "dancer": row["dancer"],
                    "gender": row["gender"],
                    "channel_id": channel_id
                })
                
                segment_id += 1
            else:
                print(f"Warning: URL {url} not found in videos_info.csv")

        print("Segments info loaded")
    
    print(f"Saving results to " + output_file)

    with open(output_file, mode='w', encoding='utf-8', newline='') as f:
        fieldnames = ["video_id", "segment_id", "start_frame", "end_frame", "type", "dancer", "gender", "channel_id"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to " + output_file)