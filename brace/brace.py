from utils.download_videos import download_videos
from utils.export_frames_from_videos import export_frames_from_segments
from utils.create_annotated_segments_csv_from_npz import check_npz_files_and_create_csv, create_dance_type_csvs
from utils.create_dataset import create_dataset_with_split
from utils.check_if_data_complete import validate_data_pairs
from utils.rename_and_restructure_npz_files import rename_and_copy_npz_files
from utils.test_dataset_labels import process_images


videos_info_csv = 'BRACE_VIDEO_INFO_PATH'
segments_csv = 'BRACE_SEGMENTS_PATH'
sequences_csv = 'BRACE_SEQUENCES_PATH'
annotated_all_segments_csv = 'DESTINATION_PATH'
annotated_powermove_segments_csv = 'DESTINATION_PATH'
annotated_toprock_segments_csv = 'DESTINATION_PATH'
annotated_footwork_segments_csv = 'DESTINATION_PATH'

manual_keypoints_folder = 'MANUAL_KEYPOINTS_FOLDER'
restructured_keypoints_folder = 'DESTINATION_FOLDER_PATH'
brace_folder = 'BRACE_FOLDER_PATH'
dataset_destinaton_folder = 'DESTINATION_FOLDER'
videos_folder = 'VIDEOS_FOLDER'
frames_folder = 'FRAMES_FOLDER'

dataset_folder = 'DESTINATION_DATASET_FOLDER'
test_dataset_folder = 'DESTINATION_TEST_DATASET_FOLDER'

def prepare_data():
    create_dance_type_csvs(segments_csv, brace_folder)
    check_npz_files_and_create_csv(segments_csv, manual_keypoints_folder, brace_folder)
    rename_and_copy_npz_files(manual_keypoints_folder, restructured_keypoints_folder)

def create_data():
    download_videos(videos_info_csv, videos_folder)
    export_frames_from_segments(annotated_all_segments_csv, videos_folder, frames_folder)

def create_dataset():
    is_dataset_complete, pairs = validate_data_pairs(annotated_all_segments_csv, restructured_keypoints_folder, frames_folder)
    if is_dataset_complete:
        create_dataset_with_split(pairs, dataset_destinaton_folder, 0.2)

def test_dataset_labels():
    process_images(dataset_folder, test_dataset_folder, 100)

if __name__ == '__main__':
    prepare_data()
    create_data()
    create_dataset()
    test_dataset_labels()