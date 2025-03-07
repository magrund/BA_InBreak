import csv

def calculate_statistics(input_file):
    statistics = {
        "total_frames": 0,
        "total_sequences": 0,
        "total_dancers": 0,
        "frames_per_type": {},
        "frames_per_dancer": {},
        "frames_per_gender": {},
        "frames_per_channel": {},
        "frames_per_video_id": {}
    }

    with open(input_file, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start_frame = int(row["start_frame"])
            end_frame = int(row["end_frame"])
            frames = end_frame - start_frame + 1 

            statistics["total_frames"] += frames

            statistics["total_sequences"] += 1

            seq_type = row["type"]
            if seq_type not in statistics["frames_per_type"]:
                statistics["frames_per_type"][seq_type] = 0
            statistics["frames_per_type"][seq_type] += frames

            dancer = row["dancer"]
            if dancer not in statistics["frames_per_dancer"]:
                statistics["total_dancers"] += 1
                statistics["frames_per_dancer"][dancer] = 0
            statistics["frames_per_dancer"][dancer] += frames

            gender = row["gender"]
            if gender not in statistics["frames_per_gender"]:
                statistics["frames_per_gender"][gender] = 0
            statistics["frames_per_gender"][gender] += frames

            channel_id = row["channel_id"]
            if channel_id not in statistics["frames_per_channel"]:
                statistics["frames_per_channel"][channel_id] = 0
            statistics["frames_per_channel"][channel_id] += frames

            video_id = row["video_id"]
            if video_id not in statistics["frames_per_video_id"]:
                statistics["frames_per_video_id"][video_id] = 0
            statistics["frames_per_video_id"][video_id] += frames

    return statistics


def save_statistics_to_file(input_file ,output_file):
    statistics = calculate_statistics(input_file)

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"Total Frames: {statistics['total_frames']}\n")

        f.write(f"\nTotal Sequences: {statistics['total_sequences']}\n")

        f.write(f"\nTotal Dancers: {statistics['total_dancers']}\n")

        f.write("\nFrames per Type:\n")
        for seq_type, frames in statistics["frames_per_type"].items():
            f.write(f"{seq_type}: {frames}\n")

        f.write("\nFrames per Dancer:\n")
        for dancer, frames in statistics["frames_per_dancer"].items():
            f.write(f"{dancer}: {frames}\n")

        f.write("\nFrames per Gender:\n")
        for gender, frames in statistics["frames_per_gender"].items():
            f.write(f"{gender}: {frames}\n")

        f.write("\nFrames per Channel:\n")
        for channel_id, frames in statistics["frames_per_channel"].items():
            f.write(f"{channel_id}: {frames}\n")

        f.write("\nFrames per Video ID:\n")
        for video_id, frames in statistics["frames_per_video_id"].items():
            f.write(f"{video_id}: {frames}\n")