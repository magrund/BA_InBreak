import csv
import os

def is_overlapping(start, end, existing_start, existing_end):
    return not (end <= existing_start or start >= existing_end)

def time_to_seconds(time_str):
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

    return int(hours) * 3600 + int(minutes) * 60 + seconds + milliseconds

def add_url_to_file(urls_file_path, url):
    urls = set()
    try:
        with open(urls_file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            urls = {row[0] for row in reader}
    except FileNotFoundError:
        pass

    if url not in urls:
        with open(urls_file_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([url])
        print("URL added to " + urls_file_path)
    else:
        print("URL already in " + urls_file_path)

def add_sequence(segments_info_file_path, urls_file_path, url, start, end, dancer, gender, seq_type):
    data = []

    try:
        with open(segments_info_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            data = list(reader)
    except FileNotFoundError:
        pass

    start_seconds = time_to_seconds(start)
    end_seconds = time_to_seconds(end)

    for row in data:
        if row["url"] == url:
            existing_start_seconds = time_to_seconds(row["start"])
            existing_end_seconds = time_to_seconds(row["end"])

            if is_overlapping(start_seconds, end_seconds, existing_start_seconds, existing_end_seconds):
                print(f"Time span {start} to {end} overlaps with an existing segment for URL: {url}.")
                return

    max_segment_id = -1
    for row in data:
        if row["url"] == url:
            max_segment_id = max(max_segment_id, int(row["segment_id"]))

    new_segment_id = max_segment_id + 1

    new_row = {
        "url": url,
        "segment_id": new_segment_id,
        "start": start,
        "end": end,
        "dancer": dancer,
        "gender": gender,
        "type": seq_type
    }
    data.append(new_row)

    with open(segments_info_file_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["url", "segment_id", "start", "end", "dancer", "gender", "type"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(data)
        
    print(f"Segment added to " + segments_info_file_path)

    add_url_to_file(urls_file_path, url)