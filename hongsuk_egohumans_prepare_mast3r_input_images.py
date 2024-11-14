import os
import os.path as osp
from pathlib import Path
import glob
import shutil
import tyro

def main(our_optimization_result_dir: str = '/home/hongsuk/projects/dust3r/outputs/egohumans/2024nov13_good_cams/num_of_cams4',
         output_dir: str = '/scratch/partial_datasets/egoexo/hongsuk/egohumans/test_images_for_mast3r_2024nov13'):
    """Copy input images used for optimization to a new directory structure for MAST3R.
    
    Args:
        our_optimization_result_dir: Directory containing optimization result pickle files
        output_dir: Directory to save copied images with new structure
    """
    # Define mapping from activity name to sequence name
    big_seq_name_dict = {
        'tagging': '01_tagging',
        'legoassemble': '02_lego',
        'fencing': '03_fencing',
        'basketball': '04_basketball',
        'volleyball': '05_volleyball',  
        'badminton': '06_badminton',
        'tennis': '07_tennis',
    }

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all pickle files
    pkl_files = sorted(glob.glob(osp.join(our_optimization_result_dir, '*.pkl')))
    print(f"Found {len(pkl_files)} pickle files")

    # Process each pickle file
    total_count = 0
    for pkl_file in pkl_files:
        # Parse filename
        # Example: 019_badminton_300_cam01cam02cam05cam07.pkl
        basename = osp.basename(pkl_file)
        parts = basename.replace('.pkl', '').split('_')
        
        # Get sequence info
        seq_num = parts[0]  # e.g. '019'
        activity = parts[1]  # e.g. 'badminton'
        frame_num = parts[2]  # e.g. '300'
        cameras = parts[3]  # e.g. 'cam01cam02cam05cam07'
        
        # Split cameras string into individual cameras
        # Using list comprehension to split 'cam01cam02cam05cam07' into ['cam01', 'cam02', 'cam05', 'cam07']
        cam_name_each_length = 5
        camera_list = [cameras[i:i+cam_name_each_length] for i in range(0, len(cameras), cam_name_each_length)]
        
        # Create sequence and frame directory path using the mapping
        big_seq = big_seq_name_dict[activity]  # e.g. '06_badminton' for 'badminton'
        small_seq = f'{seq_num}_{activity}'
        frame_dir = osp.join(output_dir, big_seq, small_seq, f'frame_{int(frame_num):05d}')
        Path(frame_dir).mkdir(parents=True, exist_ok=True)
        
        # Copy images for each camera
        for cam in camera_list:
            # Construct source image path
            src_img_path = f'./data/egohumans_data/{big_seq}/{small_seq}/exo/{cam}/images/{int(frame_num) + 1:05d}.jpg'
            
            # Construct destination image path
            dst_img_name = f'{cam}_{int(frame_num):05d}.jpg'
            dst_img_path = osp.join(frame_dir, dst_img_name)
            
            # Copy image
            if osp.exists(src_img_path):
                shutil.copy2(src_img_path, dst_img_path)
                print(f"Copied {src_img_path} to {dst_img_path}")
            else:
                print(f"Warning: Source image not found: {src_img_path}")
        
        total_count += 1
        # print(f"Copied {count} images for frame {frame_num} in {small_seq}")

    print(f"Finished copying {total_count} frames to {output_dir}")

if __name__ == '__main__':
    tyro.cli(main)