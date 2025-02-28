import os
import shutil

def rename_and_copy_npz_files(manual_keypoints_folder, restructured_keypoints_folder):
    print(f"Starting to copy files from {manual_keypoints_folder} to {restructured_keypoints_folder}")

    if not os.path.exists(restructured_keypoints_folder):
        os.makedirs(restructured_keypoints_folder)

    for root, dirs, files in os.walk(manual_keypoints_folder):
        for file in files:
            if file.endswith('.npz'):
                src_file_path = os.path.join(root, file)
                relative_path = os.path.relpath(root, manual_keypoints_folder)
                parts = relative_path.split(os.sep)
                if len(parts) > 1:
                    new_file_name = f"{parts[1]}_{file}"
                else:
                    new_file_name = file
                dest_file_path = os.path.join(restructured_keypoints_folder, new_file_name)
                
                if not os.path.exists(dest_file_path):
                    shutil.copy2(src_file_path, dest_file_path)
                    print(f"Copied {src_file_path} to {dest_file_path}")
                else:
                    print(f"File {dest_file_path} already exists. Skipping.")

    print("Finished copying files.")