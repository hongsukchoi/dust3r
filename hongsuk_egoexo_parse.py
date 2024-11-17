import os
from pathlib import Path
import glob
import json
def get_sequence_paths():
    # Base directories
    base_mvopti_dir = '/scratch/partial_datasets/egoexo/egoexo4d_v2_mvopti/run_08/val'
    base_preprocess_dir = '/scratch/partial_datasets/egoexo/preprocess_20241110_camera_ready/takes'
    
    sequence_data = []
    
    # Iterate through all sequences in the mvopti directory
    for sequence_path in glob.glob(os.path.join(base_mvopti_dir, '*/*')):
        # Get sequence name and frame number
        sequence_name = os.path.basename(os.path.dirname(sequence_path))
        frame_number = os.path.basename(sequence_path)
        
        # Construct paths
        vitpose_gt_path = os.path.join(sequence_path, 'input_data.pkl')
        
        # Construct dust3r paths following the pattern
        dust3r_base = os.path.join(base_preprocess_dir, sequence_name, 
                                 'preprocessing/dust3r_world_env_2', f'{int(frame_number):06d}', 'images')
        dust3r_network_output = os.path.join(dust3r_base, 'dust3r_network_output_pointmaps_images.pkl')
        dust3r_ga_output = os.path.join(dust3r_base, 'dust3r_global_alignment_results.pkl')
        
        # Verify that files exist
        if all(os.path.exists(p) for p in [vitpose_gt_path, dust3r_network_output, dust3r_ga_output]):
            sequence_data.append({
                'sequence_name': sequence_name,
                'frame_number': frame_number,
                'vitpose_gt_path': vitpose_gt_path,
                'dust3r_network_output_path': dust3r_network_output,
                'dust3r_ga_output_path': dust3r_ga_output
            })
        else:
            print(f"Warning: Some files missing for sequence {sequence_name}, frame {frame_number}")
    
    return sequence_data

if __name__ == "__main__":
    sequences = get_sequence_paths()
    
    # Print found sequences
    for seq in sequences:
        print("\nSequence:", seq['sequence_name'])
        print("Frame:", seq['frame_number'])
        print("VitPose GT Path:", seq['vitpose_gt_path'])
        print("Dust3r Network Output:", seq['dust3r_network_output_path'])
        print("Dust3r GA Output:", seq['dust3r_ga_output_path'])
        print("-" * 80)

    with open('egoexo_sequences.json', 'w') as f:
        json.dump(sequences, f)