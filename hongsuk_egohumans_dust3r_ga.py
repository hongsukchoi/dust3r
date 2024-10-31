"""
Total Output Data Structure:

total_output = {
    'sequence_frame_cam01cam02cam03cam04': {  # e.g. '001_tagging_0_cam01cam02cam03cam04'
        'gt_cameras': {
            'cam01': {
                'cam2world_R': np.ndarray,  # shape (3, 3), rotation matrix
                'cam2world_t': np.ndarray,  # shape (3,), translation vector
                'K': np.ndarray,  # shape (3, 3), intrinsic matrix
                'img_width': int,
                'img_height': int
            },
            'cam02': {...},
            'cam03': {...},
            'cam04': {...}
        },
        'dust3r_ga': {  # Global alignment results
            'cam01': {
                'rgbimg': np.ndarray,  # shape (H, W, 3), RGB image
                'intrinsic': np.ndarray,  # shape (3, 3), camera intrinsic matrix
                'cam2world': np.ndarray,  # shape (4, 4), camera extrinsic matrix
                'pts3d': np.ndarray,  # shape (N, 3), 3D points
                'depths': np.ndarray,  # shape (H, W), depth map
                'msk': np.ndarray,  # shape (H, W), mask
                'conf': np.ndarray,  # shape (N,), confidence scores
            },
            'cam02': {...},
            'cam03': {...},
            'cam04': {...}
        }
    },
    'sequence_frame_cam01cam02cam03cam04': {...},
    ...
}
"""

import os
import numpy as np
import copy
import pickle
import PIL
import tyro
import tqdm
import pytz
import torch

from datetime import datetime
from pathlib import Path

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.optim_factory import adjust_learning_rate_by_lr
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt.init_im_poses import init_minimum_spanning_tree, init_from_known_poses_hongsuk, init_from_pts3d
from dust3r.cloud_opt.commons import cosine_schedule, linear_schedule

from hongsuk_egohumans_dataloader import create_dataloader
from hongsuk_vis_viser_env_only import show_env_in_viser


def adjust_lr(cur_iter, niter, lr_base, lr_min, optimizer, schedule):
    t = cur_iter / niter
    if schedule == 'cosine':
        lr = cosine_schedule(t, lr_base, lr_min)
    elif schedule == 'linear':
        lr = linear_schedule(t, lr_base, lr_min)
    else:
        raise ValueError(f'bad lr {schedule=}')
    adjust_learning_rate_by_lr(optimizer, lr)
    return lr

def parse_to_save_data(scene, cam_names):
    # Get optimized values from scene
    pts3d = scene.get_pts3d()
    depths = scene.get_depthmaps()
    msk = scene.get_masks()
    confs = [c for c in scene.im_conf]
    intrinsics = scene.get_intrinsics()
    cams2world = scene.get_im_poses()

    # Convert to numpy arrays
    intrinsics = to_numpy(intrinsics)
    cams2world = to_numpy(cams2world)
    pts3d = to_numpy(pts3d)
    depths = to_numpy(depths)
    msk = to_numpy(msk)
    confs = to_numpy(confs)
    rgbimg = scene.imgs

    # Save the results as a pickle file
    results = {}
    for i, cam_name in enumerate(cam_names):
        results[cam_name] = {
            'rgbimg': rgbimg[i],
            'intrinsic': intrinsics[i],
            'cam2world': cams2world[i],
            'pts3d': pts3d[i],
            'depths': depths[i],
            'msk': msk[i],
            'conf': confs[i],
        }
    return results

def main(output_path: str = './outputs/egohumans', dust3r_output_path: str = '/home/hongsuk/projects/dust3r/outputs/egohumans/dust3r_network_output_30:11:10.pkl', egohumans_data_root: str = '/home/hongsuk/projects/egohumans/data', vis: bool = False):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # EgoHumans data
    # Fix batch size to 1 for now
    cam_names = sorted(['cam01', 'cam02', 'cam03', 'cam04'])
    dataset, dataloader = create_dataloader(egohumans_data_root, dust3r_output_path=dust3r_output_path, batch_size=1, split='test', subsample_rate=10, cam_names=cam_names)

    # Dust3r Config for the global alignment
    mode = GlobalAlignerMode.PointCloudOptimizer if len(cam_names) > 2 else GlobalAlignerMode.PairViewer
    device = 'cuda'
    silent = False
    schedule = 'linear'
    niter = 300
    lr = 0.01
    lr_base = lr
    lr_min = 0.0001
    init = 'mst'
    has_human_cue = False
    norm_scale = True
    niter_PnP = 10
    min_conf_thr_for_pnp = 3

    total_output = {}
    total_scene_num = len(dataset)
    print(f"Running global alignment for {total_scene_num} scenes")
    for i in tqdm.tqdm(range(total_scene_num), total=total_scene_num):
        sample = dataset.get_single_item(i)
        output = sample['dust3r_network_output']

        scene = global_aligner(output, device=device, mode=mode, verbose=not silent, has_human_cue=has_human_cue)

        # initialize the scene parameters with the known poses or point clouds
        if init == 'mst':
            scene.init_default_mst(niter_PnP=niter_PnP, min_conf_thr=min_conf_thr_for_pnp)
            print("Default MST init")

        # define the adam optimizer
        params = [p for p in scene.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.9))

        # Given the number of iterations, run the optimizer while forwarding the scene with the current parameters to get the loss
        with tqdm.tqdm(total=niter) as bar:
            while bar.n < bar.total:
                lr = adjust_lr(bar.n, niter, lr_base, lr_min, optimizer, schedule)
                optimizer.zero_grad()
                loss = scene.dust3r_loss()
                loss.backward()
                optimizer.step()

                bar.set_postfix_str(f'{lr=:g} loss={loss:g}')
                bar.update()
        print("final loss: ", loss)

        # Save output
        output_name = f"{sample['sequence']}_{sample['frame']}_{''.join(cam_names)}"
        total_output[output_name] = {}
        total_output[output_name]['gt_cameras'] = sample['multiview_cameras']
        total_output[output_name]['dust3r_ga'] = parse_to_save_data(scene, cam_names)
        
        # visualize
        if vis:
            show_env_in_viser(world_env=total_output[output_name]['dust3r_ga'], world_scale_factor=5.)


    # Save total output
    # get date and time (day:hour:minute)
    # time in pacific time
    now = datetime.now(pytz.timezone('US/Pacific')).strftime("%d:%H:%M")
    with open(os.path.join(output_path, f'dust3r_ga_output_{now}.pkl'), 'wb') as f:
        pickle.dump(total_output, f)

if __name__ == '__main__':
    tyro.cli(main)