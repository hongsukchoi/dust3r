import os
import os.path as osp
from pathlib import Path
import glob
import pickle
import tyro

def main(our_optimization_result_dir: str = '/home/hongsuk/projects/dust3r/outputs/egohumans/2024nov13_good_cams/num_of_cams4',
         mast3r_result_dir: str = '/scratch/partial_datasets/egoexo/hongsuk/egohumans/test_outputs_of_mast3r_2024nov13',
         output_dir: str = '/home/hongsuk/projects/dust3r/outputs/egohumans/2024nov13_good_cams_oursdust3rmast3r'):
    """Combine our optimization results with MAST3R results.
    
    Args:
        our_optimization_result_dir: Directory containing our optimization result pickle files
        mast3r_result_dir: Directory containing MAST3R result pickle files
        output_dir: Directory to save combined results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get all MAST3R result files
    mast3r_files = sorted(glob.glob(osp.join(mast3r_result_dir, '*.pkl')))
    print(f"Found {len(mast3r_files)} MAST3R result files")

    # Process each MAST3R file
    for mast3r_file in mast3r_files:
        # Get filename
        basename = osp.basename(mast3r_file)

        # Parse filename
        # Example: 019_badminton_300_cam01cam02cam05cam07.pkl
        parts = basename.replace('.pkl', '').split('_')
        
        # Get sequence info
        seq_num = parts[0]  # e.g. '019'
        activity = parts[1]  # e.g. 'badminton'
        frame_num = parts[2]  # e.g. '300'
        cameras = parts[3]  # e.g. 'cam01cam02cam05cam07'
        
        activity_name_matcher = {
            'tagging': 'tagging',
            'lego': 'legoassemble', 
            'fencing': 'fencing',
            'basketball': 'basketball',
            'volleyball': 'volleyball',
            'badminton': 'badminton',
            'tennis': 'tennis',
        }
        
        new_activity_name = activity_name_matcher[activity]  # e.g. 'badminton' for 'badminton'
        # TEMP
        if new_activity_name != 'legoassemble':
            continue
            
        # Construct our result filename to match MAST3R filename format
        our_basename = f'{seq_num}_{new_activity_name}_{frame_num}_{cameras}.pkl'
        
        # Find corresponding our result file
        our_file = osp.join(our_optimization_result_dir, our_basename)
        
        if not osp.exists(our_file):
            print(f"Warning: No matching our result file for {basename}")
            continue
        
        # Load both files
        with open(mast3r_file, 'rb') as f:
            mast3r_result = pickle.load(f)
            
        with open(our_file, 'rb') as f:
            our_result = pickle.load(f)
            
        # Add MAST3R result to our result dictionary
        our_result['mast3r_pred_world_cameras_and_structure'] = mast3r_result
        
        # Save combined result
        output_file = osp.join(output_dir, our_basename)
        with open(output_file, 'wb') as f:
            pickle.dump(our_result, f)
            
        print(f"Saved combined result to {output_file}")

    print(f"Finished combining results in {output_dir}")

if __name__ == '__main__':
    tyro.cli(main) 