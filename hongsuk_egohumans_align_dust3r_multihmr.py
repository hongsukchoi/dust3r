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
import os.path as osp
import numpy as np
import copy
import pickle
import PIL
import PIL.ImageOps as ImageOps
import cv2
import tyro
import tqdm
import pytz
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf

from collections import defaultdict
from datetime import datetime
from pathlib import Path

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.optim_factory import adjust_learning_rate_by_lr
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt.init_im_poses import init_minimum_spanning_tree, init_from_known_poses_hongsuk, init_from_pts3d
from dust3r.cloud_opt.commons import cosine_schedule, linear_schedule

from multihmr.blocks import SMPL_Layer
from multihmr.utils import get_smplx_joint_names

from hongsuk_egohumans_dataloader import create_dataloader
from hongsuk_joint_names import COCO_WHOLEBODY_KEYPOINTS, SMPLX_JOINT_NAMES

from hongsuk_vis_viser_env_only import show_env_in_viser


def draw_2d_keypoints(img, keypoints, keypoints_name=None, color=(0, 255, 0), radius=3):
    for i, keypoint in enumerate(keypoints):
        img = cv2.circle(img, (int(keypoint[0]), int(keypoint[1])), radius, color, -1)
    return img
# Get the PIL image from the dust3r torch tensor image
# revert this transform: ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
def img_to_pil(img):
    img = (img + 1) / 2
    img = tvf.ToPILImage()(img)
    return img

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

def get_resume_info(results, device='cuda'):
    pts3d = [torch.from_numpy(results[img_name]['pts3d']).to(device) for img_name in results.keys()]
    im_focals = [results[img_name]['intrinsic'][0][0] for img_name in results.keys()]
    im_poses = [torch.from_numpy(results[img_name]['cam2world']).to(device) for img_name in results.keys()]
    im_poses = torch.stack(im_poses)

    return pts3d, im_focals, im_poses

def project_points(world2cam_4by4, intrinsics, points, device='cuda'):
    # world2cam_4by4: (N, 4, 4), intrinsics: (N, 3, 3), points: (1, J, 3)
    # project the points to the image plane
    points_homo = torch.cat((points, torch.ones((1, points.shape[1], 1), device=device)), dim=2) # (1, J, 4)
    points_cam = world2cam_4by4 @ points_homo.permute(0, 2, 1) # (N, 4, J)
    points_img = intrinsics @ points_cam[:, :3, :] # (N, 3, J)
    points_img = points_img[:, :2, :] / points_img[:, 2:3, :] # (N, 2, J)
    points_img = points_img.permute(0, 2, 1) # (N, J, 2)
    return points_img

def get_human_loss(smplx_layer, humans_optim_target_dict, cam_names, multiview_world2cam_4by4, multiview_intrinsics, multiview_multiperson_poses2d, multiview_multiperson_bboxes, device='cuda'):
    # multiview_multiperson_poses2d: Dict[human_name -> Dict[cam_name -> (J, 3)]]
    # multiview_multiperson_bboxes: Dict[human_name -> Dict[cam_name -> (5)]]
    # multiview_world2cam_4by4: (N, 4, 4), multiview_intrinsics: (N, 3, 3)
    # num_cams can be different from the number of cameras (N) in the scene because some humans might not be detected in some cameras

    # save the 2D joints for visualization
    projected_joints = defaultdict(dict) # Dict[cam_name -> Dict[human_name -> (J, 3)]]

    # define different loss per view; factors are the inverse of the area of the bbox and the human detection score
    human_loss = 0
    for human_name, optim_target_dict in humans_optim_target_dict.items():
        # get the 2D joints in the image plane and the loss weights per joint
        multiview_poses2d = multiview_multiperson_poses2d[human_name] # Dict[cam_name -> (J, 3)]
        multiview_bboxes = multiview_multiperson_bboxes[human_name] # Dict[cam_name -> (5)]

        # extract data from the optim_target_dict
        transl, global_rotvec, relative_rotvec, shape, expression = optim_target_dict['transl'], optim_target_dict['global_rotvec'], optim_target_dict['relative_rotvec'], optim_target_dict['shape'], optim_target_dict['expression']

        # decode the smpl mesh and joints
        pose = torch.cat((global_rotvec, relative_rotvec), dim=1) # (1, 53, 3)
        smplx_output = smplx_layer(transl=transl,
                                 pose=pose,
                                 shape=shape,
                                 K=torch.zeros((len(pose), 3, 3), device=device),  # dummy
                                 expression=expression,
                                 loc=None,
                                 dist=None)
        smplx_j3d = smplx_output['j3d'] # (1, J, 3), joints in the world coordinate from the world mesh decoded by the optimizing parameters

        multiview_poses2d_refactored = []
        multiview_loss_weights = []
        multiview_bbox_areas = 0
        sampled_cam_indices = []
        for cam_name, bbox in multiview_bboxes.items():
            bbox_area = bbox[2] * bbox[3]
            det_score = bbox[4]
            multiview_loss_weights.append(det_score / bbox_area)
            multiview_bbox_areas += bbox_area
            sampled_cam_indices.append(cam_names.index(cam_name))
            multiview_poses2d_refactored.append(multiview_multiperson_poses2d[human_name][cam_name])
        multiview_loss_weights = torch.stack(multiview_loss_weights) # * multiview_bbox_areas   # (num_cams,)
        multiview_poses2d = torch.stack(multiview_poses2d_refactored) # (num_cams, J, 3)

        # project the joints to different views
        multiview_multihmr_j2d = project_points(multiview_world2cam_4by4[sampled_cam_indices], multiview_intrinsics[sampled_cam_indices], smplx_j3d, device=device) # (num_cams, J, 2)

        # map the multihmr 2d pred to the COCO_WHOLEBODY_KEYPOINTS
        multiview_multihmr_j2d_transformed = torch.zeros(len(multiview_multihmr_j2d), len(COCO_WHOLEBODY_KEYPOINTS), 3, device=device)
        for i, joint_name in enumerate(COCO_WHOLEBODY_KEYPOINTS):
            if joint_name in SMPLX_JOINT_NAMES:
                multiview_multihmr_j2d_transformed[:, i, :2] = multiview_multihmr_j2d[:, SMPLX_JOINT_NAMES.index(joint_name), :2]
                multiview_multihmr_j2d_transformed[:, i, 2] = 1 # for validity check. 1 if the joint is valid, 0 otherwise

        multiview_multihmr_j2d_transformed[:, :COCO_WHOLEBODY_KEYPOINTS.index('right_heel'), 2] *= 100 # main body joints are weighted 10 times more

        # compute the hubor loss using Pytorch between multiview_multihmr_j2d_transformed and multiview_poses2d
        one_human_loss = multiview_loss_weights[:, None, None].repeat(1, multiview_multihmr_j2d_transformed.shape[1], 1) \
        * multiview_multihmr_j2d_transformed[:, :, 2:] * multiview_poses2d[:, :, 2:] \
        * F.smooth_l1_loss(multiview_multihmr_j2d_transformed[:, :, :2], multiview_poses2d[:, :, :2], reduction='none').mean(dim=-1, keepdim=True)
  
        human_loss += one_human_loss.mean()

        # for visualization purpose
        for idx, sam_cam_idx in enumerate(sampled_cam_indices):
            projected_joints[cam_names[sam_cam_idx]][human_name] = multiview_multihmr_j2d_transformed[idx]

    return human_loss, projected_joints

def init_human_params(multihmr_output, device = 'cuda'):
    # multihmr_output: dict of human parameters

    optim_target_dict = {} # human_name: str -> Dict[param_name: str -> nn.Parameter]
    for human_name, smplx_3d_params in multihmr_output.items():
        # TEMP
        if human_name != 'aria01':
            continue

        # first extract data from the dictionary of smplx_3d_params 
        transl, rotvec, shape, expression = smplx_3d_params['transl'], smplx_3d_params['rotvec'], smplx_3d_params['shape'], smplx_3d_params['expression']
        # transl: (3), rotvec: (53, 3), shape: (10), expression: (10)
        global_rotvec = rotvec[0:1, :] 
        relative_rotvec = rotvec[1:, :]

        # initialize transl, rotvec, shape, expression as network parameters that have gradients
        optim_target_dict[human_name] = {}
        optim_target_dict[human_name]['transl'] = nn.Parameter(transl.unsqueeze(0).to(device)) # (1, 3)
        optim_target_dict[human_name]['global_rotvec'] = nn.Parameter(global_rotvec.unsqueeze(0).to(device)) # (1, 1, 3)
        optim_target_dict[human_name]['relative_rotvec'] = nn.Parameter(relative_rotvec.unsqueeze(0).to(device)) # (1, 52, 3)
        optim_target_dict[human_name]['shape'] = nn.Parameter(shape.unsqueeze(0).to(device)) # (1, 10)
        optim_target_dict[human_name]['expression'] = nn.Parameter(expression.unsqueeze(0).to(device)) # (1, 10)

        # TEMP
        # make relative_rotvec, shape, expression not require grad
        optim_target_dict[human_name]['relative_rotvec'].requires_grad = False
        optim_target_dict[human_name]['shape'].requires_grad = False
        optim_target_dict[human_name]['expression'].requires_grad = False

    return optim_target_dict


# TODO
# Make sure to use GPU
# (optional) optimize the separate scale paramter for the dust3r world
#  get groundtruth human poses and match them with the multihmr output
# Visaulize the 2D projected joints and the GT joints

# Debug steps:
# First, check if rerunning global alignment with scaled 3D points and camera poses gives the same results as the previous global alignment
# Second, initialize the camera parameters with GT and initialize the human translation with GT and see if it projects correctly
# I think there might be indexing error around camera names and human names


def main(output_path: str = './outputs/egohumans', multihmr_output_path: str = '/home/hongsuk/projects/dust3r/outputs/egohumans/multihmr_output_30:23:17.pkl', dust3r_ga_output_path: str = '/home/hongsuk/projects/dust3r/outputs/egohumans/dust3r_ga_output_30:17:54.pkl', dust3r_output_path: str = '/home/hongsuk/projects/dust3r/outputs/egohumans/dust3r_network_output_30:11:10.pkl', egohumans_data_root: str = '/home/hongsuk/projects/egohumans/data', vis: bool = False):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # EgoHumans data
    # Fix batch size to 1 for now
    cam_names = sorted(['cam01', 'cam02', 'cam03', 'cam04'])
    dataset, dataloader = create_dataloader(egohumans_data_root, dust3r_output_path=dust3r_output_path, dust3r_ga_output_path=dust3r_ga_output_path, multihmr_output_path=multihmr_output_path, batch_size=1, split='test', subsample_rate=10, cam_names=cam_names)

    # Dust3r Config for the global alignment
    mode = GlobalAlignerMode.PointCloudOptimizer if len(cam_names) > 2 else GlobalAlignerMode.PairViewer
    device = 'cuda'
    silent = False
    schedule = 'linear'
    niter = 300
    lr = 0.01
    lr_base = lr
    lr_min = 0.0001
    init = 'known_params_hongsuk'
    norm_scale = True
    niter_PnP = 10
    min_conf_thr_for_pnp = 3
    norm_pw_scale = False

    # define different learning rate for the human parameters
    human_lr = lr * 1.0
    smplx_layer = SMPL_Layer(type='smplx', gender='neutral', num_betas=10, kid=False, person_center='head')
    smplx_layer = smplx_layer.to(device)
    save_2d_pose_vis = 20 

    total_output = {}
    total_scene_num = len(dataset)
    print(f"Running global alignment for {total_scene_num} scenes")
    for i in tqdm.tqdm(range(total_scene_num), total=total_scene_num):
        sample = dataset.get_single_item(i)

        """ Initialize the scene parameters """
        # load the dust3r network output
        dust3r_network_output = sample['dust3r_network_output']

        # load the precomputed 3D points, camera poses, and intrinsics from Dust3r GA output
        dust3r_ga_output = sample['dust3r_ga_output']
        pts3d, im_focals, im_poses = get_resume_info(dust3r_ga_output, device)

        # TEMP
        # do the optimization again with scaled 3D points and camera poses
        scale = 100.
        pts3d_scaled = [p * scale for p in pts3d]
        pts3d = pts3d_scaled
        im_poses[:, :3, 3] = im_poses[:, :3, 3] * scale

        # define the scene class that will be optimized
        scene = global_aligner(dust3r_network_output, device=device, mode=mode, verbose=not silent, has_human_cue=False)
        scene.norm_pw_scale = norm_pw_scale


        # initialize the scene parameters with the known poses or point clouds
        if init == 'mst':
            scene.init_default_mst(niter_PnP=niter_PnP, min_conf_thr=min_conf_thr_for_pnp)
            print("Default MST init")
        elif init == 'known_params_hongsuk':
            scene.init_from_known_params_hongsuk(im_focals=im_focals, im_poses=im_poses, pts3d=pts3d, niter_PnP=niter_PnP, min_conf_thr=min_conf_thr_for_pnp)
            print("Known params init")

        scene_params = [p for p in scene.parameters() if p.requires_grad]

        """ Initialize the human parameters """
        # load the human data from MultiHMR output
        multihmr_output = sample['multihmr_output'] # dict of human parameters

        # load the 2D joints in the image plane and the loss weights per joint
        # multiview_multiple_human_2d_cam_annot[camera_name][human_name] = {
        #     'pose2d': pose2d,
        #     'bbox': bbox
        # }
        multiview_multiperson_annots = sample['multiview_multiple_human_2d_cam_annot'] # Dict[camera_name -> Dict[human_name -> (J, 3)]]
        multiview_affine_transforms = sample['multiview_affine_transforms'] # Dict[camera_name -> np.ndarray]
        multiview_images = sample['multiview_images'] # Dict[camera_name -> Dict] # for visualization
        multiview_multiperson_poses2d = defaultdict(dict)
        multiview_multiperson_bboxes = defaultdict(dict)
        # make a dict of human_name -> Dict[cam_name -> (J, 3)]
        # make a dict of human_name -> Dict[cam_name -> (5)] for bboxes; xywh confidence
        for cam_name in multiview_multiperson_annots.keys():
            for human_name in multiview_multiperson_annots[cam_name].keys():
                pose2d = multiview_multiperson_annots[cam_name][human_name]['pose2d']
                pose2d[:, 2] = 1
                bbox = multiview_multiperson_annots[cam_name][human_name]['bbox']

                # affine transform the 2D joints
                multiview_affine_transform = multiview_affine_transforms[cam_name]
                # make it to 3by3 matrix
                multiview_affine_transform = np.concatenate((multiview_affine_transform, np.array([[0, 0, 1]])), axis=0)
                # get inverse; original image space to dust3r image space
                multiview_affine_transform = np.linalg.inv(multiview_affine_transform)
                pose2d = multiview_affine_transform @ pose2d.T
                pose2d = pose2d.T
                pose2d[:, 2] = multiview_multiperson_annots[cam_name][human_name]['pose2d'][:, 2] # confidence

                # affine transform the bbox
                reshaped_bbox = np.array([bbox[0], bbox[1], 1, bbox[2], bbox[3], 1.]).reshape(2, 3)
                reshaped_bbox = multiview_affine_transform @ reshaped_bbox.T
                reshaped_bbox = reshaped_bbox.T
                bbox[:4] = reshaped_bbox[:2, :2].reshape(4)

                if vis:
                    img = multiview_images[cam_name]['img']
                    img = img_to_pil(img)
                    img = np.array(img)
                    img = draw_2d_keypoints(img, pose2d)
                    # draw the bbox
                    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
                    cv2.imwrite(osp.join(output_path, f'{sample["sequence"]}_{sample["frame"]}_{cam_name}_{human_name}_2d_keypoints_bbox.png'), img[..., ::-1])

                multiview_multiperson_poses2d[human_name][cam_name] = torch.from_numpy(pose2d).to(device)
                multiview_multiperson_bboxes[human_name][cam_name] = torch.from_numpy(bbox).to(device)


        # initialize the human paramters to optimize
        human_params_to_optimize = []
        human_params_names_to_optimize = []
        human_params = init_human_params(multihmr_output, device) # dict of human parameters
        for human_name, optim_target_dict in human_params.items():
            for param_name in optim_target_dict.keys():
                if optim_target_dict[param_name].requires_grad:
                    human_params_to_optimize.append(optim_target_dict[param_name])
                    human_params_names_to_optimize.append(f'{human_name}_{param_name}')
        print(f"Optimizing {len(human_params_names_to_optimize)} parameters of humans: {human_params_names_to_optimize}")

        # Visualize the initilization of human and world
        # # TEMP
        # from hongsuk_vis_viser_env_human import show_env_human_in_viser
        # smplx_3d_params = multihmr_output['aria01']
        # smplx_output = smplx_layer(transl=smplx_3d_params['transl'][None, :],
        #                     pose=smplx_3d_params['rotvec'][None, :],
        #                     shape=smplx_3d_params['shape'][None, :],
        #                     K=torch.zeros((len(smplx_3d_params['rotvec']), 3, 3), device=device),  # dummy
        #                     expression=smplx_3d_params['expression'][None, :],
        #                     loc=None,
        #     dist=None)
        # smplx_vertices_dict = {
        #     'world': smplx_output['v3d'].detach().squeeze().cpu().numpy()
        # }
        # show_env_human_in_viser(world_env=sample['dust3r_ga_output'], world_scale_factor=20., smplx_vertices_dict=smplx_vertices_dict, smplx_faces=smplx_layer.bm_x.faces)

        # Create parameter groups with different learning rates
        param_groups = [
            {'params': scene_params, 'lr': lr},
            {'params': human_params_to_optimize, 'lr': human_lr}
        ]

        # Initialize Adam optimizer with parameter groups
        optimizer = torch.optim.Adam(param_groups, betas=(0.9, 0.9))

        # Given the number of iterations, run the optimizer while forwarding the scene with the current parameters to get the loss
        with tqdm.tqdm(total=niter) as bar:
            while bar.n < bar.total:
                lr = adjust_lr(bar.n, niter, lr_base, lr_min, optimizer, schedule)
                optimizer.zero_grad()
                scene_loss = scene.dust3r_loss()

                # get extrinsincs and intrinsics from the scene
                multiview_cam2world_4by4  = scene.get_im_poses()  # (len(cam_names), 4, 4)
                multiview_world2cam_4by4 = torch.inverse(multiview_cam2world_4by4) # (len(cam_names), 4, 4)
                multiview_intrinsics = scene.get_intrinsics() # (len(cam_names), 3, 3)

                # get human loss
                human_loss, projected_joints = get_human_loss(smplx_layer, human_params, cam_names, multiview_world2cam_4by4, multiview_intrinsics, multiview_multiperson_poses2d, multiview_multiperson_bboxes, device)
                loss = scene_loss + human_loss * 15

                loss.backward()
                optimizer.step()

                bar.set_postfix_str(f'{lr=:g} loss={loss:g}, scene_loss={scene_loss:g}, human_loss={human_loss:g}')
                bar.update()

                if bar.n % save_2d_pose_vis == 0:
                    for cam_name, human_joints in projected_joints.items():
                        img = scene.imgs[cam_names.index(cam_name)].copy() * 255.
                        img = img.astype(np.uint8)
                        for human_name, joints in human_joints.items():
                            for joint in joints:
                                img = cv2.circle(img, (int(joint[0]), int(joint[1])), 3, (0, 255, 0), -1)
                        cv2.imwrite(osp.join(output_path, f'{sample["sequence"]}_{sample["frame"]}_{cam_name}_{human_name}_{bar.n}.png'), img[:, :, ::-1])
                    
        print("final losses: ", scene_loss.item(), human_loss.item())

        # Save output
        output_name = f"{sample['sequence']}_{sample['frame']}_{''.join(cam_names)}"
        total_output[output_name] = {}
        total_output[output_name]['gt_cameras'] = sample['multiview_cameras']
        total_output[output_name]['dust3r_ga'] = parse_to_save_data(scene, cam_names)
        
        show_env_in_viser(world_env=total_output[output_name]['dust3r_ga'], world_scale_factor=5.)


    # Save total output
    # get date and time (day:hour:minute)
    # time in pacific time
    now = datetime.now(pytz.timezone('US/Pacific')).strftime("%d:%H:%M")
    with open(os.path.join(output_path, f'dust3r_ga_output_{now}.pkl'), 'wb') as f:
        pickle.dump(total_output, f)

if __name__ == '__main__':
    tyro.cli(main)