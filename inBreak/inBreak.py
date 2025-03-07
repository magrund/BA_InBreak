from utils.add_segments_to_segments_info import add_sequence
from utils.create_video_info_csv import create_video_info_csv
from utils.create_segment_info_csv import create_video_segments_csv
from utils.calculate_statistics_from_data import save_statistics_to_file
from utils.download_videos_from_youtube import download_video_segments
from utils.export_frames_from_video import export_frames_from_segments
from utils.check_if_all_frames_exported import check_if_all_frames_exist_from_csv
from utils.mediapipe_annotation.annotate_videos_with_mediapipe import annotate_segments_in_folder

urls_path = 'dataset/urls.csv'
segments_info_path = 'dataset/segments_info.csv'
video_info_path = 'dataset/video_info.csv'
video_segments_path = 'dataset/video_segments.csv'
statistics_path = 'dataset/statistics.txt'

video_segments_folder = '/Volumes/4TB_Podonne/BA_Marcel_Grund/inBreak/segments'
frames_folder = '/Volumes/4TB_Podonne/BA_Marcel_Grund/inBreak/frames'
annotations_folder = '/Volumes/4TB_Podonne/BA_Marcel_Grund/inBreak/annotations'


def add_data():
    add_sequence(
        segments_info_path,
        urls_path,
        url="https://www.youtube.com/watch?v=kLHb01dBVpk",
        start="00:01:43.000", # HH:MM:SS.mmm
        end="00:01:44.000", # HH:MM:SS.mmm
        dancer="MakuGold",
        gender="female", # male / female
        seq_type="Transition" # Transition / Powermove / Freeze
    )

def create_data_info():
    create_video_info_csv(urls_path, video_info_path)
    create_video_segments_csv(video_info_path, segments_info_path, video_segments_path)

def create_statistics():
    save_statistics_to_file(video_segments_path, statistics_path)

def createData():
    download_video_segments(video_info_path, segments_info_path, video_segments_folder)
    export_frames_from_segments(video_segments_path, video_segments_folder, frames_folder)
    check_if_all_frames_exist_from_csv(video_segments_path, frames_folder)

def createAutomatedAnnotation():
    annotate_segments_in_folder(frames_folder, annotations_folder)

if __name__ == '__main__':
    mode = "createData"
    if mode == "addData":
        add_data()
    elif mode == "createData":
        create_data_info()
        create_statistics()
        createData()
        createAutomatedAnnotation()