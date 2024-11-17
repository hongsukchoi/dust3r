import os
import shutil
from pathlib import Path
from tqdm import tqdm
def copy_with_structure():
    # Source and destination base paths
    src_base = '/scratch/one_month/2024_10/lmueller/egohuman/camera_ready'
    dst_base = '/scratch/partial_datasets/bedlam/egohumans/extracted/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready'

    # Walk through the source directory
    for root, dirs, files in tqdm(os.walk(src_base)):
        # Get the relative path from the source base
        rel_path = os.path.relpath(root, src_base)
        
        if 'dust3r_world_env' in rel_path or 'images_frame100' in rel_path or 'identified_predictions' in rel_path:
            continue
        
        # Replace 'humanwithhand' with the new name in the path
        new_rel_path = rel_path.replace('humanwithhand', 'lea_pred_hmr2_hamer_vit_bytetrack')

        # Construct destination path
        dst_path = os.path.join(dst_base, new_rel_path)
        
        # Create destination directory if it doesn't exist
        Path(dst_path).mkdir(parents=True, exist_ok=True)
        
        # Copy all files in the current directory
        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(dst_path, file)
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")

if __name__ == "__main__":
    copy_with_structure() 