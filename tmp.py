output_dir1 = "/home/hongsuk/projects/dust3r/outputs/egohumans/2024nov14_good_cams_focal_fixed/num_of_cams2"
output_dir2 = "/home/hongsuk/projects/dust3r/outputs/egohumans/2024nov14_good_cams_focal_fixed/num_of_cams4"

import os
import glob

# Get all files from both directories
files1 = set(os.path.basename(f).split("final_")[0] for f in glob.glob(os.path.join(output_dir1, "*.pkl")))
files2 = set(os.path.basename(f).split("final_")[0] for f in glob.glob(os.path.join(output_dir2, "*.pkl")))

# Find files that are in one directory but not the other
only_in_dir1 = files1 - files2
only_in_dir2 = files2 - files1

print("Files only in first directory:")
for f in sorted(only_in_dir1):
    print(f"  {f}")

print("\nFiles only in second directory:")
for f in sorted(only_in_dir2):
    print(f"  {f}")


