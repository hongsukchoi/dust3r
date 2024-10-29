import os
import os.path as osp
import numpy as np
import torch
import pickle
import tyro

from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from hongsuk_vis_viser_env_only import show_env_in_viser


def get_reconstructed_scene(output, device, silent, mode, schedule, niter, lr=0.01, init='mst', norm_scale=True, pts3d=None, im_focals=None, im_poses=None):
    """
    From a list of images, run global aligner and return reconstructed scene data.
    """
    
    # Select an optimizer
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent, has_human_cue=False)

    if pts3d is None:
        loss = scene.compute_global_alignment(init=init, niter=niter, schedule=schedule, lr=lr)
    else:
        if not norm_scale:
            scene.norm_pw_scale = False
        
        if init == 'known_poses_hongsuk':
            from dust3r.utils.geometry import geotrf, inv

            for i in range(scene.n_imgs):
                cam2world = im_poses[i]
                depth = geotrf(inv(cam2world), pts3d[i])[..., 2]
                scene._set_depthmap(i, depth)

                scene._set_pose(scene.im_poses, i, im_poses[i])
                if im_focals[i] is not None:
                    scene._set_focal(i, im_focals[i])
            # principal points are not optimized before and now
            print("N iter: ", niter)
            loss = scene.compute_global_alignment(init=init, niter=niter, schedule=schedule, lr=lr)
        
        elif init == 'known_pts3d':
            loss = scene.compute_global_alignment(init=init, niter=niter, schedule=schedule, lr=lr, pts3d=pts3d, im_focals=im_focals, im_poses=im_poses)
    
    print('final loss: ', loss)

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

    return rgbimg, intrinsics, cams2world, pts3d, depths, msk, confs


def get_resume_info(alignment_results_file, device):
    with open(alignment_results_file, 'rb') as f:
        results = pickle.load(f)
    
    pts3d = [torch.from_numpy(results[img_name]['pts3d']).to(device) for img_name in results.keys()]
    im_focals = [results[img_name]['intrinsic'][0][0] for img_name in results.keys()]
    im_poses = [torch.from_numpy(results[img_name]['cam2world']).to(device) for img_name in results.keys()]
    im_poses = torch.stack(im_poses)

    return pts3d, im_focals, im_poses


def main(net_pred_file: str, resume_file: str = None):
    # Load the DUSt3R network output
    with open(net_pred_file, 'rb') as f:
        total_output = pickle.load(f)
    output = total_output['output']
    img_names = total_output['img_names']

    # Config for the global alignment
    mode = GlobalAlignerMode.PointCloudOptimizer if len(img_names) > 2 else GlobalAlignerMode.PairViewer
    device = 'cuda'
    silent = False
    schedule = 'linear'
    niter = 990
    lr = 0.01
    

    # Run the global alignment
    if resume_file is None:
        init = 'mst'
        rgbimg, intrinsics, cams2world, pts3d, depths, msk, confs = get_reconstructed_scene(output, device, silent, mode, schedule, niter, lr, init)
    else:
        pts3d, im_focals, im_poses = get_resume_info(resume_file, device)
        print("Resuming global alignment from: ", resume_file)
        # init = 'known_pts3d'
        init = 'known_poses_hongsuk'
        norm_scale = False
        
        # TEMP
        # do the optimization again with scaled 3D points and camera poses
        scale = 10.
        pts3d_scaled = [p * scale for p in pts3d]
        pts3d = pts3d_scaled
        im_poses[:, :3, 3] = im_poses[:, :3, 3] * scale

        rgbimg, intrinsics, cams2world, pts3d, depths, msk, confs = get_reconstructed_scene(output, device, silent, mode, schedule, niter, lr, init, \
                                                                                            norm_scale, pts3d, im_focals, im_poses)

    # Save the results as a pickle file
    results = {}
    for i, img_name in enumerate(img_names):
        results[img_name] = {
            'rgbimg': rgbimg[i],
            'intrinsic': intrinsics[i],
            'cam2world': cams2world[i],
            'pts3d': pts3d[i],
            'depths': depths[i],
            'msk': msk[i],
            'conf': confs[i],
        }
    
    scene_name = osp.basename(net_pred_file).split('.')[0].split('_')[-1]
    dir_name = osp.dirname(net_pred_file)
    if resume_file is None:
        output_file = osp.join(dir_name, f'dust3r_global_alignment_results_{scene_name}.pkl')
    else:
        output_file = osp.join(dir_name, f'dust3r_global_alignment_results_{scene_name}_resume.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Global alignment results saved to {output_file}")

    # Show visualizations in viser
    show_env_in_viser(output_file, world_scale_factor=5.)


if __name__ == '__main__':
    tyro.cli(main)