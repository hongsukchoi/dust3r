import tyro
import pickle
import os.path as osp
import torch
import tqdm
import dust3r.cloud_opt.init_im_poses as init_fun

from dust3r.cloud_opt.init_im_poses import init_minimum_spanning_tree, init_from_known_poses_hongsuk, init_from_pts3d
from dust3r.optim_factory import adjust_learning_rate_by_lr
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from hongsuk_vis_viser_env_only import show_env_in_viser

from dust3r.cloud_opt.commons import cosine_schedule, linear_schedule


def get_resume_info(alignment_results_file, device):
    with open(alignment_results_file, 'rb') as f:
        results = pickle.load(f)
    
    pts3d = [torch.from_numpy(results[img_name]['pts3d']).to(device) for img_name in results.keys()]
    im_focals = [results[img_name]['intrinsic'][0][0] for img_name in results.keys()]
    im_poses = [torch.from_numpy(results[img_name]['cam2world']).to(device) for img_name in results.keys()]
    im_poses = torch.stack(im_poses)

    return pts3d, im_focals, im_poses

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


def main(net_pred_file: str, resume_file: str = None):
    # load the dust3r network prediction
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
    niter = 290
    lr = 0.01
    lr_base = 0.01
    lr_min = 0.0001
    init = 'mst'
    has_human_cue = False
    norm_scale = True
    niter_PnP = 10
    min_conf_thr_for_pnp = 3

    # optionally load the known poses or point clouds
    if resume_file is not None:
        pts3d, im_focals, im_poses = get_resume_info(resume_file, device)
        print("Resuming global alignment from: ", resume_file)
        # init = 'known_pts3d'
        init = 'known_params_hongsuk'
        norm_scale = False

        # TEMP
        # do the optimization again with scaled 3D points and camera poses
        scale = 10.
        pts3d_scaled = [p * scale for p in pts3d]
        pts3d = pts3d_scaled
        im_poses[:, :3, 3] = im_poses[:, :3, 3] * scale

    # run the global alignment
    # define the scene, which is actually a neural network that has learnable parameters which are the scene parameters and to be optimized
    scene = global_aligner(output, device=device, mode=mode, verbose=not silent, has_human_cue=has_human_cue)
    if norm_scale:
        scene.norm_pw_scale = True
    else:
        scene.norm_pw_scale = False

    # initialize the scene parameters with the known poses or point clouds
    if init == 'mst':
        scene.init_default_mst(niter_PnP=niter_PnP, min_conf_thr=min_conf_thr_for_pnp)
        print("Default MST init")
    elif init == 'known_params_hongsuk':
        scene.init_from_known_params_hongsuk(im_focals=im_focals, im_poses=im_poses, pts3d=pts3d, niter_PnP=niter_PnP, min_conf_thr=min_conf_thr_for_pnp)
        print("Known params init")
        

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



if __name__ == "__main__":
    tyro.cli(main)