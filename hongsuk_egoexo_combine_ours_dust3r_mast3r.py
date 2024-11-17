import os
import os.path as osp
import glob
import pickle
import re
from tqdm import tqdm

from pathlib import Path
def load_reference_results(reference_dir):
    """Load all reference results and organize them by sequence and frame."""
    reference_results = {}
    
    # Pattern to extract sequence and frame from reference filename
    ref_pattern = r'(.+?)_(\d+)_final'
    
    for filepath in glob.glob(os.path.join(reference_dir, '*.pkl')):
        filename = os.path.basename(filepath)
        match = re.match(ref_pattern, filename)
        
        if match:
            sequence_name = match.group(1)
            frame_number = match.group(2)
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
                reference_results[(sequence_name, frame_number)] = {
                    'filepath': filepath,
                    'data': data
                }
    
    return reference_results

def find_matching_mast3r_result(mast3r_dir, sequence_name, frame_number):
    """Find matching MAST3R result file based on sequence and frame number."""
    # Convert frame number to match MAST3R format (remove leading zeros)
    frame_num_int = int(frame_number)
    
    # Pattern to match MAST3R files
    pattern = f"{sequence_name}_{frame_num_int}_cam*.pkl"
    
    matching_files = glob.glob(os.path.join(mast3r_dir, pattern))
    return matching_files[0] if matching_files else None

def combine_results():
    reference_dir = '/scratch/partial_datasets/egoexo/hongsuk/egoexo/optim_outputs'
    mast3r_dir = '/scratch/partial_datasets/egoexo/hongsuk/egoexo/test_outputs_of_mast3r_2024nov14'
    output_dir = '/scratch/partial_datasets/egoexo/hongsuk/egoexo/optim_outputs/mast3r_combined'

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load all reference results
    reference_results = load_reference_results(reference_dir)
    
    # Process each reference result and find matching MAST3R result

    count = 0    
    for (sequence_name, frame_number), ref_data in tqdm(reference_results.items()):
        mast3r_filepath = find_matching_mast3r_result(mast3r_dir, sequence_name, frame_number)
        
        if mast3r_filepath:
            with open(mast3r_filepath, 'rb') as f:
                mast3r_data = pickle.load(f)
                
            # Create combined result by adding MAST3R data to reference data
            ref_data['mast3r_pred_world_cameras_and_structure'] = mast3r_data
            
            # Save combined result with original reference filename
            output_filepath = osp.join(output_dir, f"{sequence_name}_{frame_number}_final.pkl")
            with open(output_filepath, 'wb') as f:
                pickle.dump(ref_data, f)
            count += 1
            print(f"Processed: {sequence_name} - Frame {frame_number}")
        else:
            print(f"Warning: No matching MAST3R result for {sequence_name} - Frame {frame_number}")

    print("Saved total of {} files".format(count))
if __name__ == "__main__":
    combine_results()
