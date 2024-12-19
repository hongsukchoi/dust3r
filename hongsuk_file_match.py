output_dir1 = "/scratch/partial_datasets/egoexo/hongsuk/egohumans/optim_outputs/2024nov25_noscaleinit/num_of_cams4"
output_dir2 = "/home/hongsuk/projects/dust3r/outputs/egohumans/2024nov14_good_cams_focal_fixed/num_of_cams4"

import os
import glob

def get_seq_frame_name(filename):
    # Split by underscore and take first three parts (seq_activity_frame)
    parts = os.path.basename(filename).split('_')
    return '_'.join(parts[:3])

# Get all files from both directories and extract sequence_frame_names
files1 = set(get_seq_frame_name(f) for f in glob.glob(os.path.join(output_dir1, "*.pkl")))
files2 = set(get_seq_frame_name(f) for f in glob.glob(os.path.join(output_dir2, "*.pkl")))

# Find differences
only_in_dir1 = files1 - files2
only_in_dir2 = files2 - files1

print(f"Files only in num_of_cams2 ({len(only_in_dir1)}):")
for f in sorted(only_in_dir1):
    print(f"  {f}")

print(f"\nFiles only in num_of_cams4 ({len(only_in_dir2)}):")
for f in sorted(only_in_dir2):
    print(f"  {f}")

