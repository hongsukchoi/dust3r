"""
GA output Data Structure:
{ 
    'gt_cameras': {
        'cam01': {
            'cam2world_4by4': np.ndarray,  # shape (4, 4), camera extrinsic matrix
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
    },
    'img_names': (sequence, frame, cam_names)
}
"""

import os
import os.path as osp
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

def parse_to_save_data(scene, cam_names, main_cam_idx=None):
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

    if main_cam_idx is not None:
        main_cam_cam2world = cams2world[main_cam_idx]
        # transform all the cameras and pts3d with the transformation matrix, which is the inverse of the main cam extrinsic matrix
        main_cam_world2cam = np.linalg.inv(main_cam_cam2world)
        for i, cam_name in enumerate(cam_names):
            cams2world[i] = main_cam_world2cam @ cams2world[i]
            pts3d[i] =  pts3d[i] @ main_cam_world2cam[:3, :3].T + main_cam_world2cam[:3, 3:].T

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

def main(output_dir: str = './outputs/egohumans/', dust3r_raw_output_dir: str = './outputs/egohumans/dust3r_raw_outputs/2024nov13_good_cams', egohumans_data_root: str = './data/egohumans_data', vis: bool = False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # EgoHumans data
    # Fix batch size to 1 for now
    selected_big_seq_list = ['07_tennis'] #['06_badminton']#['02_lego'] #['06_badminton']  #['07_tennis'] #  # #['01_tagging', '02_lego, 05_volleyball', '04_basketball', '03_fencing'] # ##[, , ''] 
    selected_small_seq_start_and_end_idx_tuple = (6,13) #(1, 20)
    cam_names = None #sorted(['cam01', 'cam02', 'cam03', 'cam04'])
    num_of_cams = 4
    use_sam2_mask = False
    subsample_rate = 50 #100
    output_dir = osp.join(output_dir, 'dust3r_ga_outputs_and_gt_cameras', '2024nov13_good_cams', f'num_of_cams{num_of_cams}')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    dust3r_raw_output_dir = osp.join(dust3r_raw_output_dir, f'num_of_cams{num_of_cams}')
    dataset, dataloader = create_dataloader(egohumans_data_root, dust3r_raw_output_dir=dust3r_raw_output_dir, batch_size=1, split='test', subsample_rate=subsample_rate, cam_names=cam_names, num_of_cams=num_of_cams, use_sam2_mask=use_sam2_mask, selected_big_seq_list=selected_big_seq_list, selected_small_seq_start_and_end_idx_tuple=selected_small_seq_start_and_end_idx_tuple)

    # Dust3r Config for the global alignment
    mode = GlobalAlignerMode.PointCloudOptimizer if num_of_cams > 2 else GlobalAlignerMode.PairViewer
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

    total_scene_num = len(dataset)
    print(f"Running global alignment for {total_scene_num} scenes")
    for i in tqdm.tqdm(range(total_scene_num), total=total_scene_num):
        sample = dataset.get_single_item(i)
        cam_names = sorted(sample['multiview_images'].keys())

        output = sample['dust3r_network_output']
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent, has_human_cue=has_human_cue)

        # initialize the scene parameters with the known poses or point clouds
        if mode == GlobalAlignerMode.PointCloudOptimizer and init == 'mst':
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
            print("final loss: ", loss.item())

        # Save output
        output_name = f"{sample['sequence']}_{sample['frame']}"
        to_save_data = {}
        to_save_data['gt_cameras'] = sample['multiview_cameras']
        to_save_data['dust3r_ga'] = parse_to_save_data(scene, cam_names, 0)
        to_save_data['img_names'] = (sample['sequence'], sample['frame'], cam_names)

        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{output_name}.pkl')
        print(f'Saving output to {output_path}')
        with open(output_path, 'wb') as f:
            pickle.dump(to_save_data, f)

        # visualize
        if vis:
            try:
                show_env_in_viser(world_env=to_save_data['dust3r_ga'], world_scale_factor=10., gt_cameras=to_save_data['gt_cameras'])
            except:
                import pdb; pdb.set_trace()


if __name__ == '__main__':
    tyro.cli(main)