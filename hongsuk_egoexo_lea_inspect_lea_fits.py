import os
import pickle
import viser
import viser.transforms as vtf
import numpy as np 
import argparse
import smplx 
import time
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--result-folder-base', type=str,
        default='/scratch/partial_datasets/egoexo/egoexo4d_v2_mvopti/debug_01/train'
    )
    parser.add_argument(
        '--take-name', type=str, 
        default='unc_basketball_03-31-23_02_19'
    )
    parser.add_argument(
        '--run', type=str, 
        default='debug_001'
    )
    parser.add_argument(
        '--frame', type=str,
        default='54'
    )
    parser.add_argument(
        '--port', type=str, 
        default='8097'
    )
    args = parser.parse_args()
    return args

def load_fitted_results(result_folder_base, take_name, run, frame):
    result_folder_run = f'{result_folder_base}/{take_name}/{run}'
    fitted_result_path = f'{result_folder_run}/{frame}/results/prefit.pkl'
    fitted_result = pickle.load(open(fitted_result_path, 'rb'))
    return fitted_result

def viz_fitted_results(fitted_result, port='8097'):
    """
    fitted_result: dict with output from human+cam fitting
    """

    # create viser server
    server = viser.ViserServer(port=port)

    # create smplx model (to get the faces)
    bm = smplx.create(
        model_path='/home/hongsuk/projects/egoexo/essentials/body_models', 
        model_type='smplx'
    )

    # add camera estimate 
    fitted_cams = fitted_result['cam']['T']
    for cam_idx, cam_pose in enumerate(fitted_cams):
        server.scene.add_batched_axes(
            f'fitted-camera-{cam_idx}', 
            batched_wxyzs=np.array(
                # Convert Nx3x3 rotation matrices to Nx4 quaternions.
                [vtf.SO3.from_matrix(cam_pose[:3, :3]).wxyz]
            ),
            batched_positions=np.array([cam_pose[:3, 3]]),
            axes_length=1.0,
            axes_radius=0.1,
        )

    # add dust3r points
    dust3r_points = fitted_result['scene']
    server.scene.add_point_cloud(
        f'dust3r-scene-scaled',
        points=dust3r_points['point_cloud'],
        colors=dust3r_points['point_cloud_colors'],
        point_size=0.1,
        point_shape='rounded'
    )

    # add the human meshes 
    fitted_meshes = fitted_result['human']['vertices']
    zero_position = np.eye(4)
    for mesh_idx, mesh in enumerate(fitted_meshes):
        server.scene.add_mesh_simple(
            f'fitted-human-{mesh_idx}',
            vertices=mesh[0],
            faces=bm.faces.astype(int),
            position=zero_position[:3, 3],
            wxyz=vtf.SO3.from_matrix(zero_position[:3, :3]).wxyz,
        )

    # set break point for debugging
    # import ipdb; ipdb.set_trace()
    # Add GUI elements.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    start_time = time.time()
    while True:
        time.sleep(0.01)
        timing_handle.value = (time.time() - start_time) 

def main():

    args = parse_args()
    result_folder_base = args.result_folder_base
    take_name = args.take_name
    run = args.run
    frame=args.frame
    port=args.port

    fitted_result = load_fitted_results(
        result_folder_base, take_name, run, frame
    )

    viz_fitted_results(fitted_result, port=port)

if __name__ == '__main__':
    main()