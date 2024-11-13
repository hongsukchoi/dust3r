output_dir1 = "/home/hongsuk/projects/dust3r/outputs/egoexo/nov11/sota_comparison_trial1"
output_dir2 = "/home/hongsuk/projects/dust3r/outputs/egoexo/nov11/sota_comparison_trial1_use_gt_focal"

import os
import glob

# Get all files from both directories
files1 = set(os.path.basename(f).split("final_")[0] for f in glob.glob(os.path.join(output_dir1, "*final_*.pkl")))
files2 = set(os.path.basename(f).split("final_")[0] for f in glob.glob(os.path.join(output_dir2, "*final_*.pkl")))
import pdb; pdb.set_trace()
# Find files that are in one directory but not the other
only_in_dir1 = files1 - files2
only_in_dir2 = files2 - files1

print("Files only in first directory:")
for f in sorted(only_in_dir1):
    print(f"  {f}")

print("\nFiles only in second directory:")
for f in sorted(only_in_dir2):
    print(f"  {f}")


