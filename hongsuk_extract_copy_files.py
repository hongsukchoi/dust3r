"""
read text file that contains the list of file indicators to copy like this:
01_tagging, 1 1
01_tagging, 2 2
01_tagging, 3 3
01_tagging, 4 4
01_tagging, 5 5
01_tagging, 6 6
01_tagging, 7 7
01_tagging, 8 8
01_tagging, 9 9
01_tagging, 10 10
01_tagging, 11 11
01_tagging, 12 12
01_tagging, 13 13
01_tagging, 14 14
02_lego, 1 1
02_lego, 2 2
02_lego, 3 3
02_lego, 4 4
02_lego, 5 5
02_lego, 6 6
03_fencing, 1 1
03_fencing, 2 2
03_fencing, 3 3
03_fencing, 4 4
03_fencing, 5 5
...


The corresponding files are in the following directory:
/scratch/partial_datasets/egoexo/hongsuk/egohumans/optim_outputs/2024nov19/num_of_cams4_num_of_humans1

The file names are like this:

001_tagging_0_cam01cam04cam06cam08.pkl
001_tagging_100_cam01cam04cam06cam08.pkl
001_tagging_200_cam01cam04cam06cam08.pkl
001_tagging_300_cam01cam04cam06cam08.pkl
...

The files start with 001_tagging correspond to 01_tagging, 1 1
The files start with 002_tagging correspond to 01_tagging, 2 2
...

Copy the corresponding files to the following directory:
/scratch/partial_datasets/egoexo/hongsuk/egohumans/optim_outputs/2024nov19/num_of_cams4_num_of_humans1_to_evaluate

"""

import os
import shutil
from pathlib import Path

def read_file_indicators(file_path):
    """Read the text file containing file indicators and return a list of tuples."""
    indicators = []
    with open(file_path, 'r') as f:
        for line in f:
            # Split the line and remove any whitespace
            parts = [part.strip() for part in line.split(',')]
            if len(parts) == 2:
                scene_name = parts[0]
                numbers = parts[1].split()
                if len(numbers) == 2:
                    indicators.append((scene_name, int(numbers[0]), int(numbers[1])))
    return indicators

def get_corresponding_filename_pattern(scene_name, sequence_num):
    """Convert scene name and sequence number to the corresponding filename format.
    For example: 
    '01_tagging', 1 -> '001_tagging_0_cam01cam04cam06cam08.pkl'
    '01_tagging', 2 -> '002_tagging_0_cam01cam04cam06cam08.pkl'
    """
    # Get the scene type (tagging, lego, etc.)
    _, scene_type = scene_name.split('_')
    if scene_type == 'lego':
        scene_type = 'legoassemble'
    # Format the sequence number to 3 digits (1 -> 001, 2 -> 002, etc.)
    formatted_scene = f"{sequence_num:03d}_{scene_type}"
    return f"{formatted_scene}_*_*.pkl"

def copy_files():
    # Define source and destination directories
    # src_dir = Path("/scratch/partial_datasets/egoexo/hongsuk/egohumans/optim_outputs/2024nov19/num_of_cams4_num_of_humans4")
    # dst_dir = Path("/scratch/partial_datasets/egoexo/hongsuk/egohumans/optim_outputs/2024nov19/num_of_cams4_num_of_humans4_to_evaluate")
    src_dir = Path("/scratch/partial_datasets/egoexo/hongsuk/egohumans/optim_outputs/2024nov25/num_of_cams4_num_of_humans0")
    dst_dir = Path("/scratch/partial_datasets/egoexo/hongsuk/egohumans/optim_outputs/2024nov25/num_of_cams4_num_of_humans0_to_evaluate")

    # Create destination directory if it doesn't exist
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    # Read the indicators file (assuming it's in the same directory as the script)
    script_dir = Path(__file__).parent
    indicators_file = script_dir / "file_indicators.txt" # _num_cams
    
    # Get the list of files to copy
    indicators = read_file_indicators(indicators_file)
    
    # Copy each file
    for scene_name, num1, num2 in indicators:
        filename_pattern = get_corresponding_filename_pattern(scene_name, num1)
        src_files = sorted(list(src_dir.glob(filename_pattern)))
        dst_files = [dst_dir / file.name for file in src_files]
        
        if src_files:
            try:
                for src_file, dst_file in zip(src_files, dst_files):
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied: {src_file}")
            except Exception as e:
                print(f"Error copying {filename_pattern}: {str(e)}")
        else:
            print(f"Source file not found: {filename_pattern}")

if __name__ == "__main__":
    copy_files()