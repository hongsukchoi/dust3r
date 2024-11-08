"""
# Data structure for total_output:
total_output = {
    'world_gt_cameras': {  # Ground truth camera parameters in world coordinates
        'cam01': {
            'cam2world_R': np.ndarray,  # shape (3, 3), rotation matrix
            'cam2world_t': np.ndarray,  # shape (3,), translation vector
            'K': np.ndarray,  # shape (3, 3), intrinsic matrix
            'img_width': int,
            'img_height': int,
            'cam2world_4by4': np.ndarray  # shape (4, 4), homogeneous transformation matrix
        },
        'cam02': {...},
        'cam03': {...},
        'cam04': {...}
    },
    'world_multiple_human_3d_annot': {  # Ground truth 3D human parameters
        'human_name': {  # e.g. 'aria01'
            'body_pose': np.ndarray,  # SMPL body pose parameters
            'betas': np.ndarray,  # SMPL shape parameters
            'global_orient': np.ndarray,  # Global orientation
            'root_transl': np.ndarray,  # Root translation
            'transl': np.ndarray  # Translation
        },
        'human_name2': {...}
    },
    'dust3r_ga': {  # Global alignment results from DUSt3R
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
    'human_params': {  # Optimized human parameters
        'human_name': {  # e.g. 'aria01' 
            'body_pose': np.ndarray,  # shape (1, 63), SMPL body pose parameters
            'global_orient': np.ndarray,  # shape (1, 3), global orientation
            'betas': np.ndarray,  # shape (1, 10), SMPL shape parameters
            'left_hand_pose': np.ndarray,  # shape (1, 45), left hand pose parameters
            'right_hand_pose': np.ndarray,  # shape (1, 45), right hand pose parameters
            'root_transl': np.ndarray  # shape (1, 3), root translation
        },
        'human_name2': {...}
    }
}

>> How to decode GT SMPL parameters
    body_pose = torch.from_numpy(human_params['body_pose']).reshape(1, -1).to(device).float()
    global_orient = torch.from_numpy(human_params['global_orient']).reshape(1, -1).to(device).float()
    betas = torch.from_numpy(human_params['betas']).reshape(1, -1).to(device).float()

    smpl_output = smpl_layer(betas=betas,
                            body_pose=body_pose,
                            global_orient=global_orient,
                            pose2rot=True,
                        )
    root_transl = human_params['root_transl'] # np.ndarray (1, 3)
    vertices = smpl_output.vertices.detach().squeeze(0).cpu().numpy()
    joints = smpl_output.joints.detach().squeeze(0).cpu().numpy()
    vertices = vertices - joints[0:1:, ] + root_transl
    world_human_vertices[human_name] = vertices

>> How to decode HMR2Hamer SMPL-X parameters
# extract data from the optim_target_dict
body_pose = optim_target_dict['body_pose'].reshape(1, -1)
betas = optim_target_dict['betas'].reshape(1, -1)
global_orient = optim_target_dict['global_orient'].reshape(1, -1)
left_hand_pose = optim_target_dict['left_hand_pose'].reshape(1, -1)
right_hand_pose = optim_target_dict['right_hand_pose'].reshape(1, -1)

# decode the smpl mesh and joints
smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)

# Add root translation to the joints
root_transl = optim_target_dict['root_transl'].reshape(1, 1, -1)
smplx_j3d = smplx_output.joints # (1, J, 3), joints in the world coordinate from the world mesh decoded by the optimizing parameters
smplx_j3d = smplx_j3d - smplx_j3d[:, 0:1, :] + root_transl # !ALWAYS! Fuck the params['transl']

# If you are applying rotation to the global orientation, you have to always compensate the rotation of the root joint translation
"""

import os
import os.path as osp
import numpy as np
import copy
import time
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
import smplx
import warnings

from typing import List
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.optim_factory import adjust_learning_rate_by_lr
from dust3r.utils.device import to_numpy
from dust3r.cloud_opt.init_im_poses import init_minimum_spanning_tree, init_from_known_poses_hongsuk, init_from_pts3d
from dust3r.cloud_opt.commons import cosine_schedule, linear_schedule

from hongsuk_vis_viser_env_only import visualize_cameras, procrustes_align
from hongsuk_vis_viser_env_human import visualize_cameras_and_human, show_env_human_in_viser

from hongsuk_egohumans_dataloader import create_dataloader
from hongsuk_joint_names import COCO_WHOLEBODY_KEYPOINTS, ORIGINAL_SMPLX_JOINT_NAMES

coco_main_body_end_joint_idx = COCO_WHOLEBODY_KEYPOINTS.index('right_heel') 
coco_main_body_joint_idx = list(range(coco_main_body_end_joint_idx + 1))
coco_main_body_joint_names = COCO_WHOLEBODY_KEYPOINTS[:coco_main_body_end_joint_idx + 1]
smplx_main_body_joint_idx = [ORIGINAL_SMPLX_JOINT_NAMES.index(joint_name) for joint_name in coco_main_body_joint_names] 

# Define skeleton edges using indices of main body joints
COCO_MAIN_BODY_SKELETON = [
    # Torso
    [5, 6],   # left_shoulder to right_shoulder
    [5, 11],  # left_shoulder to left_hip
    [6, 12],  # right_shoulder to right_hip
    [11, 12], # left_hip to right_hip
    
    # Left arm
    [5, 7],   # left_shoulder to left_elbow
    [7, 9],   # left_elbow to left_wrist
    
    # Right arm
    [6, 8],   # right_shoulder to right_elbow
    [8, 10],  # right_elbow to right_wrist
    
    # Left leg
    [11, 13], # left_hip to left_knee
    [13, 15], # left_knee to left_ankle
    [15, 19], # left_ankle to left_heel
    
    # Right leg
    [12, 14], # right_hip to right_knee
    [14, 16], # right_knee to right_ankle
    [16, 22], # right_ankle to right_heel

    # Head
    [0, 1], # nose to left_eye
    [0, 2], # nose to right_eye
    [1, 3], # left_eye to left_ear
    [2, 4], # right_eye to right_ear
]


def draw_2d_keypoints(img, keypoints, keypoints_name=None, color=(0, 255, 0), radius=1):
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

class Timer:
    def __init__(self):
        self.times = []
        self.start_time = None

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        if self.start_time is None:
            raise RuntimeError("Timer.tic() must be called before Timer.toc()")
        self.times.append(time.time() - self.start_time)
        self.start_time = None

    @property
    def average_time(self):
        if not self.times:
            return 0.0
        return sum(self.times) / len(self.times)
    
    @property
    def total_time(self):
        return sum(self.times)

def get_resume_info(results, device='cuda'):
    cam_names = sorted(list(results.keys()))
    pts3d = [torch.from_numpy(results[img_name]['pts3d']).to(device) for img_name in cam_names]
    im_focals = [results[img_name]['intrinsic'][0][0] for img_name in cam_names]
    im_poses = [torch.from_numpy(results[img_name]['cam2world']).to(device) for img_name in cam_names]
    im_poses = torch.stack(im_poses)

    return pts3d, im_focals, im_poses, cam_names

def project_points(world2cam_4by4, intrinsics, points, device='cuda'):
    # world2cam_4by4: (N, 4, 4), intrinsics: (N, 3, 3), points: (1, J, 3)
    # project the points to the image plane
    points_homo = torch.cat((points, torch.ones((1, points.shape[1], 1), device=device)), dim=2) # (1, J, 4)
    points_cam = world2cam_4by4 @ points_homo.permute(0, 2, 1) # (N, 4, J)
    points_img = intrinsics @ points_cam[:, :3, :] # (N, 3, J)
    points_img = points_img[:, :2, :] / points_img[:, 2:3, :] # (N, 2, J)
    points_img = points_img.permute(0, 2, 1) # (N, J, 2)
    return points_img

def get_prev_human_loss(smplx_layer, humans_optim_target_dict, cam_names, multiview_world2cam_4by4, multiview_intrinsics, multiview_multiperson_poses2d, multiview_multiperson_bboxes, shape_prior_weight=0, device='cuda'):
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
        body_pose = optim_target_dict['body_pose'].reshape(1, -1)
        betas = optim_target_dict['betas'].reshape(1, -1)
        global_orient = optim_target_dict['global_orient'].reshape(1, -1)
        left_hand_pose = optim_target_dict['left_hand_pose'].reshape(1, -1)
        right_hand_pose = optim_target_dict['right_hand_pose'].reshape(1, -1)

        # decode the smpl mesh and joints
        smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)

        # Add root translation to the joints
        root_transl = optim_target_dict['root_transl'].reshape(1, 1, -1)
        smplx_j3d = smplx_output.joints # (1, J, 3), joints in the world coordinate from the world mesh decoded by the optimizing parameters
        smplx_j3d = smplx_j3d - smplx_j3d[:, 0:1, :] + root_transl # !ALWAYS! Fuck the params['transl']

        # get the joint loss weight factors
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
        multiview_loss_weights = torch.stack(multiview_loss_weights).float() # * multiview_bbox_areas   # (num_cams,)
        multiview_poses2d = torch.stack(multiview_poses2d_refactored).float() # (num_cams, J, 3)

        # project the joints to different views
        multiview_smplx_j2d = project_points(multiview_world2cam_4by4[sampled_cam_indices], multiview_intrinsics[sampled_cam_indices], smplx_j3d, device=device) # (num_cams, J, 2)

        # map the multihmr 2d pred to the COCO_WHOLEBODY_KEYPOINTS
        multiview_smplx_j2d_coco_ordered = torch.zeros(len(multiview_smplx_j2d), len(COCO_WHOLEBODY_KEYPOINTS), 3, device=device, dtype=torch.float32)
        for i, joint_name in enumerate(COCO_WHOLEBODY_KEYPOINTS):
            if joint_name in ORIGINAL_SMPLX_JOINT_NAMES:
                multiview_smplx_j2d_coco_ordered[:, i, :2] = multiview_smplx_j2d[:, ORIGINAL_SMPLX_JOINT_NAMES.index(joint_name), :2]
                multiview_smplx_j2d_coco_ordered[:, i, 2] = 1 # for validity check. 1 if the joint is valid, 0 otherwise

        multiview_smplx_j2d_coco_ordered[:, :COCO_WHOLEBODY_KEYPOINTS.index('right_heel')+1, 2] *= 100 # main body joints are weighted 10 times more
        # multiview_multihmr_j2d_transformed[:, COCO_WHOLEBODY_KEYPOINTS.index('right_heel')+1:, 2] = 0 # ignore non-main body joints

        # compute the hubor loss using Pytorch between multiview_multihmr_j2d_transformed and multiview_poses2d
        # one_human_loss = multiview_loss_weights[:, None, None].repeat(1, multiview_smplx_j2d_coco_ordered.shape[1], 1) \
        # * multiview_smplx_j2d_coco_ordered[:, :, 2:] * multiview_poses2d[:, :, 2:] \
        # * F.smooth_l1_loss(multiview_smplx_j2d_coco_ordered[:, :, :2], multiview_poses2d[:, :, :2], reduction='none').mean(dim=-1, keepdim=True)

        # TEMP; just use main keypoints
        multiview_smplx_j2d_coco_ordered = multiview_smplx_j2d_coco_ordered[:, coco_main_body_joint_idx, :]
        multiview_poses2d = multiview_poses2d[:, coco_main_body_joint_idx, :]
        # Compute the l2 
        one_human_loss = multiview_loss_weights[:, None, None].repeat(1, multiview_smplx_j2d_coco_ordered.shape[1], 1) \
        * multiview_smplx_j2d_coco_ordered[:, :, 2:] * multiview_poses2d[:, :, 2:] \
        * F.mse_loss(multiview_smplx_j2d_coco_ordered[:, :, :2], multiview_poses2d[:, :, :2], reduction='none').mean(dim=-1, keepdim=True)

        human_loss += one_human_loss.mean()
        if shape_prior_weight > 0:
            # L2 loss to regularize the shape vector
            human_loss += shape_prior_weight * F.mse_loss(betas, torch.zeros_like(betas))

        # for visualization purpose
        for idx, sam_cam_idx in enumerate(sampled_cam_indices):
            projected_joints[cam_names[sam_cam_idx]][human_name] = multiview_smplx_j2d_coco_ordered[idx]

    return human_loss, projected_joints


def get_human_loss(smplx_layer_dict, humans_optim_target_dict, cam_names, multiview_world2cam_4by4, multiview_intrinsics, multiview_multiperson_poses2d, multiview_multiperson_bboxes, shape_prior_weight=0, device='cuda'):
    # multiview_multiperson_poses2d: Dict[human_name -> Dict[cam_name -> (J, 3)]]
    # multiview_multiperson_bboxes: Dict[human_name -> Dict[cam_name -> (5)]]
    # multiview_world2cam_4by4: (N, 4, 4), multiview_intrinsics: (N, 3, 3)

    # save the 2D joints for visualization
    projected_joints = defaultdict(dict)

    # Collect all human parameters into batched tensors
    human_names = list(humans_optim_target_dict.keys())
    batch_size = len(human_names)

    # # define the smplx layer
    # smplx_layer = smplx.create(model_path = '/home/hongsuk/projects/egoexo/essentials/body_models', model_type = 'smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, use_face_contour = True, num_betas = 10, batch_size = batch_size).to(device)
    smplx_layer = smplx_layer_dict[batch_size]

    # Batch all SMPL parameters
    body_pose = torch.cat([humans_optim_target_dict[name]['body_pose'].reshape(1, -1) for name in human_names], dim=0)
    betas = torch.cat([humans_optim_target_dict[name]['betas'].reshape(1, -1) for name in human_names], dim=0)
    global_orient = torch.cat([humans_optim_target_dict[name]['global_orient'].reshape(1, -1) for name in human_names], dim=0)
    left_hand_pose = torch.cat([humans_optim_target_dict[name]['left_hand_pose'].reshape(1, -1) for name in human_names], dim=0)
    right_hand_pose = torch.cat([humans_optim_target_dict[name]['right_hand_pose'].reshape(1, -1) for name in human_names], dim=0)
    root_transl = torch.cat([humans_optim_target_dict[name]['root_transl'].reshape(1, 1, -1) for name in human_names], dim=0)

    # Forward pass through SMPL-X model for all humans at once
    smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)

    # Add root translation to joints
    smplx_j3d = smplx_output.joints  # (B, J, 3)
    smplx_j3d = smplx_j3d - smplx_j3d[:, 0:1, :] + root_transl  # (B, J, 3)

    # Project joints to all camera views at once
    # Reshape for batch projection
    B, J, _ = smplx_j3d.shape
    N = len(cam_names)
    
    # Expand camera parameters to match batch size
    world2cam_expanded = multiview_world2cam_4by4.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, 4, 4)
    intrinsics_expanded = multiview_intrinsics.unsqueeze(0).expand(B, -1, -1, -1)  # (B, N, 3, 3)
    
    # Expand joints to match number of cameras
    smplx_j3d_expanded = smplx_j3d.unsqueeze(1).expand(-1, N, -1, -1)  # (B, N, J, 3)
    
    # Project all joints at once
    points_homo = torch.cat((smplx_j3d_expanded, torch.ones((B, N, J, 1), device=device)), dim=3)  # (B, N, J, 4)
    points_cam = torch.matmul(world2cam_expanded, points_homo.transpose(2, 3))  # (B, N, 4, J)
    points_img = torch.matmul(intrinsics_expanded, points_cam[:, :, :3, :])  # (B, N, 3, J)
    points_img = points_img[:, :, :2, :] / points_img[:, :, 2:3, :]  # (B, N, 2, J)
    points_img = points_img.transpose(2, 3)  # (B, N, J, 2)

    # Initialize total loss
    total_loss = 0

    # Process each human's loss in parallel
    for human_idx, human_name in enumerate(human_names):
        # Get camera indices and loss weights for this human
        cam_indices = []
        loss_weights = []
        poses2d = []
        bbox_areas = 0
        
        for cam_name, bbox in multiview_multiperson_bboxes[human_name].items():
            bbox_area = bbox[2] * bbox[3]
            det_score = bbox[4]
            loss_weights.append(det_score / bbox_area)
            bbox_areas += bbox_area
            cam_indices.append(cam_names.index(cam_name))
            poses2d.append(multiview_multiperson_poses2d[human_name][cam_name])

        loss_weights = torch.stack(loss_weights).float().to(device)
        poses2d = torch.stack(poses2d).float().to(device)  # (num_cams, J, 3)

        # Get projected joints for this human
        human_proj_joints = points_img[human_idx, cam_indices]  # (num_cams, J, 2)

        # Create COCO ordered joints
        human_proj_joints_coco = torch.zeros(len(cam_indices), len(COCO_WHOLEBODY_KEYPOINTS), 3, device=device, dtype=torch.float32)
        for i, joint_name in enumerate(COCO_WHOLEBODY_KEYPOINTS):
            if joint_name in ORIGINAL_SMPLX_JOINT_NAMES:
                human_proj_joints_coco[:, i, :2] = human_proj_joints[:, ORIGINAL_SMPLX_JOINT_NAMES.index(joint_name), :2]
                human_proj_joints_coco[:, i, 2] = 1

        # Weight main body joints more heavily
        human_proj_joints_coco[:, :COCO_WHOLEBODY_KEYPOINTS.index('right_heel')+1, 2] *= 100

        # Get only main body keypoints
        human_proj_joints_coco = human_proj_joints_coco[:, coco_main_body_joint_idx, :]
        poses2d = poses2d[:, coco_main_body_joint_idx, :]

        # Compute MSE loss with weights
        one_human_loss = loss_weights[:, None, None].repeat(1, human_proj_joints_coco.shape[1], 1) \
            * human_proj_joints_coco[:, :, 2:] * poses2d[:, :, 2:] \
            * F.mse_loss(human_proj_joints_coco[:, :, :2], poses2d[:, :, :2], reduction='none').mean(dim=-1, keepdim=True)

        total_loss += one_human_loss.mean()

        # Store projected joints for visualization
        for idx, cam_idx in enumerate(cam_indices):
            projected_joints[cam_names[cam_idx]][human_name] = human_proj_joints_coco[idx]

    # Add shape prior if requested
    if shape_prior_weight > 0:
        total_loss += shape_prior_weight * F.mse_loss(betas, torch.zeros_like(betas))

    return total_loss, projected_joints

def estimate_initial_trans(joints3d, joints2d, focal, princpt, skeleton):
    """
    use focal length and bone lengths to approximate distance from camera
    joints3d: (J, 3), xyz in meters
    joints2d: (J, 2+1), xy pixels + confidence
    focal: scalar
    princpt: (2,), x, y
    skeleton: list of edges (bones)

    returns:
        init_trans: (3,), x, y, z in meters, translation vector of the pelvis (root) joint
    """
    # Calculate bone lengths and confidence for each bone
    bone3d_array = []  # 3D bone lengths in meters
    bone2d_array = []  # 2D bone lengths in pixels
    conf2d = []        # Confidence scores for each bone

    for edge in skeleton:
        # 3D bone length
        joint1_3d = joints3d[edge[0]]
        joint2_3d = joints3d[edge[1]]
        bone_length_3d = np.linalg.norm(joint1_3d - joint2_3d)
        bone3d_array.append(bone_length_3d)

        # 2D bone length
        joint1_2d = joints2d[edge[0], :2]  # xy coordinates
        joint2_2d = joints2d[edge[1], :2]  # xy coordinates
        bone_length_2d = np.linalg.norm(joint1_2d - joint2_2d)
        bone2d_array.append(bone_length_2d)

        # Confidence score for this bone (minimum of both joint confidences)
        bone_conf = min(joints2d[edge[0], 2], joints2d[edge[1], 2])
        conf2d.append(bone_conf)

    # Convert to numpy arrays
    bone3d_array = np.array(bone3d_array)
    bone2d_array = np.array(bone2d_array)
    conf2d = np.array(conf2d)

    mean_bone3d = np.mean(bone3d_array, axis=0)
    mean_bone2d = np.mean(bone2d_array * (conf2d > 0.0), axis=0)

    # Estimate z using the ratio of 3D to 2D bone lengths
    # z = f * (L3d / L2d) where f is focal length
    z = mean_bone3d / mean_bone2d * focal
    
    # Find pelvis (root) joint position in 2D
    pelvis_2d = joints2d[0, :2]  # Assuming pelvis is the first joint

    # Back-project 2D pelvis position to 3D using estimated z
    x = (pelvis_2d[0] - princpt[0]) * z / focal
    y = (pelvis_2d[1] - princpt[1]) * z / focal

    init_trans = np.array([x, y, z])
    return init_trans

def init_human_params(smplx_layer, multiview_multiple_human_cam_pred, multiview_multiperson_pose2d, focal_length, princpt, device = 'cuda', get_vertices=False):
    # multiview_multiple_human_cam_pred: Dict[camera_name -> Dict[human_name -> 'pose2d', 'bbox', 'params' Dicts]]
    # multiview_multiperson_pose2d: Dict[human_name -> Dict[cam_name -> (J, 2+1)]] torch tensor
    # focal_length: scalar, princpt: (2,), device: str

    # Initialize Stage 1: Get the 3D root translation of all humans from all cameras
    # Decode the smplx mesh and get the 3D bone lengths / compare them with the bone lengths from the vitpose 2D bone lengths
    camera_names = sorted(list(multiview_multiple_human_cam_pred.keys()))
    first_cam = camera_names[0]
    first_cam_human_name_counts = {human_name: 0 for human_name in multiview_multiple_human_cam_pred[first_cam].keys()}
    missing_human_names_in_first_cam = defaultdict(list)
    multiview_multiperson_init_trans = defaultdict(dict) # Dict[human_name -> Dict[cam_name -> (3)]]
    for cam_name in camera_names:
        for human_name in multiview_multiple_human_cam_pred[cam_name].keys():
            params = multiview_multiple_human_cam_pred[cam_name][human_name]['params']
            body_pose = params['body_pose'].reshape(1, -1).to(device)
            global_orient = params['global_orient'].reshape(1, -1).to(device)
            betas = params['betas'].reshape(1, -1).to(device)
            left_hand_pose = params['left_hand_pose'].reshape(1, -1).to(device)
            right_hand_pose = params['right_hand_pose'].reshape(1, -1).to(device)
            transl = params['transl'].reshape(1, -1).to(device)
            
            smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=transl)

            # Extract main body joints and visualize 3D skeleton from SMPL-X
            smplx_joints = smplx_output['joints']
            # Save the root joint (pelvis) translation for later compensation
            params['org_cam_root_transl'] = smplx_joints[0, 0,:3].detach().cpu().numpy()

            smplx_coco_main_body_joints = smplx_joints[0, smplx_main_body_joint_idx, :].detach().cpu().numpy()
            vitpose_2d_keypoints = multiview_multiperson_pose2d[human_name][cam_name][coco_main_body_joint_idx].cpu().numpy() # (J, 2+1)
            init_trans = estimate_initial_trans(smplx_coco_main_body_joints, vitpose_2d_keypoints, focal_length, princpt, COCO_MAIN_BODY_SKELETON)
            # How to use init_trans?
            # vertices = vertices - joints[0:1] + init_trans # !ALWAYS! Fuck the params['transl']
            if human_name in first_cam_human_name_counts.keys():
                first_cam_human_name_counts[human_name] += 1
            else:
                missing_human_names_in_first_cam[human_name].append(cam_name)
            multiview_multiperson_init_trans[human_name][cam_name] = init_trans

    # main human is the one that is detected in the first camera and has the most detections across all cameras
    main_human_name = None
    max_count = 0
    for human_name, count in first_cam_human_name_counts.items():
        if count == len(camera_names):
            main_human_name = human_name
            max_count = len(camera_names)
            break
        elif count > max_count:
            max_count = count
            main_human_name = human_name
    if max_count != len(camera_names):
        print(f"Warning: {main_human_name} is the most detected main human but not detected in all cameras")
    
    # Initialize Stage 2: Get the initial camera poses with respect to the first camera
    global_orient_first_cam = multiview_multiple_human_cam_pred[first_cam][main_human_name]['params']['global_orient'][0].cpu().numpy()
    # axis angle to rotation matrix
    global_orient_first_cam = R.from_rotvec(global_orient_first_cam).as_matrix().astype(np.float32)
    init_trans_first_cam = multiview_multiperson_init_trans[main_human_name][first_cam]

    # First camera (world coordinate) pose
    world_T_first = np.eye(4, dtype=np.float32)  # Identity rotation and zero translation

    # Calculate other camera poses relative to world (first camera)
    cam_poses = {first_cam: world_T_first}  # Store all camera poses
    for cam_name in multiview_multiperson_init_trans[main_human_name].keys():
        if cam_name == first_cam:
            continue
        
        # Get human orientation and position in other camera
        global_orient_other_cam = multiview_multiple_human_cam_pred[cam_name][main_human_name]['params']['global_orient'][0].cpu().numpy()
        global_orient_other_cam = R.from_rotvec(global_orient_other_cam).as_matrix().astype(np.float32)
        init_trans_other_cam = multiview_multiperson_init_trans[main_human_name][cam_name]

        # The human's orientation should be the same in world coordinates
        # Therefore: R_other @ global_orient_other = R_first @ global_orient_first
        # Solve for R_other: R_other = (R_first @ global_orient_first) @ global_orient_other.T
        R_other = global_orient_first_cam @ global_orient_other_cam.T

        # For translation: The human position in world coordinates should be the same when viewed from any camera
        # world_p = R_first @ p_first + t_first = R_other @ p_other + t_other
        # Since R_first = I and t_first = 0:
        # p_first = R_other @ p_other + t_other
        # Solve for t_other: t_other = p_first - R_other @ p_other
        t_other = init_trans_first_cam - R_other @ init_trans_other_cam

        # Create 4x4 transformation matrix
        T_other = np.eye(4, dtype=np.float32)
        T_other[:3, :3] = R_other
        T_other[:3, 3] = t_other

        cam_poses[cam_name] = T_other

    # Visualize the camera poses (cam to world (first cam))
    # visualize_cameras(cam_poses)

    # Now cam_poses contains all camera poses in world coordinates
    # The poses can be used to initialize the scene

    # Organize the data for optimization
    # Get the first cam human parameters with the initial translation
    first_cam_human_params = {}
    for human_name in multiview_multiple_human_cam_pred[first_cam].keys():
        first_cam_human_params[human_name] = multiview_multiple_human_cam_pred[first_cam][human_name]['params']
        first_cam_human_params[human_name]['root_transl'] = torch.from_numpy(multiview_multiperson_init_trans[human_name][first_cam]).reshape(1, -1).to(device)

    # Initialize Stage 3: If the first camera (world coordinate frame) has missing person,
    # move other camera view's human to the first camera view's human's location
    for missing_human_name in missing_human_names_in_first_cam:
        missing_human_exist_cam_idx = 0
        other_cam_name = missing_human_names_in_first_cam[missing_human_name][missing_human_exist_cam_idx]
        while other_cam_name not in cam_poses.keys():
            missing_human_exist_cam_idx += 1
            if missing_human_exist_cam_idx == len(missing_human_names_in_first_cam[missing_human_name]):
                print(f"Warning: {missing_human_name} cannot be handled because it can't transform to the first camera coordinate frame")
                continue
            other_cam_name = missing_human_names_in_first_cam[missing_human_name][missing_human_exist_cam_idx]
        missing_human_params_in_other_cam = multiview_multiple_human_cam_pred[other_cam_name][missing_human_name]['params']
        # keys: 'body_pose', 'betas', 'global_orient', 'right_hand_pose', 'left_hand_pose', 'transl'
        # transform the missing_human_params_in_other_cam to the first camera coordinate frame
        other_cam_to_first_cam_transformation = cam_poses[other_cam_name] # (4,4)
        missing_human_params_in_other_cam_global_orient = missing_human_params_in_other_cam['global_orient'][0].cpu().numpy() # (3,)
        missing_human_params_in_other_cam_global_orient = R.from_rotvec(missing_human_params_in_other_cam_global_orient).as_matrix().astype(np.float32) # (3,3)
        missing_human_params_in_other_cam_global_orient = other_cam_to_first_cam_transformation[:3, :3] @ missing_human_params_in_other_cam_global_orient # (3,3)
        missing_human_params_in_other_cam['global_orient'] = torch.from_numpy(R.from_matrix(missing_human_params_in_other_cam_global_orient).as_rotvec().astype(np.float32)).to(device) # (3,)

        missing_human_init_trans_in_other_cam = multiview_multiperson_init_trans[missing_human_name][other_cam_name]
        missing_human_init_trans_in_first_cam = other_cam_to_first_cam_transformation[:3, :3] @ missing_human_init_trans_in_other_cam + other_cam_to_first_cam_transformation[:3, 3]
        # compenstate rotation (translation from origin to root joint was not cancled)
        root_transl_compensator = other_cam_to_first_cam_transformation[:3, :3] @ missing_human_params_in_other_cam['org_cam_root_transl'] 
        missing_human_init_trans_in_first_cam = missing_human_init_trans_in_first_cam + root_transl_compensator
        #
        missing_human_params_in_other_cam['root_transl'] = torch.from_numpy(missing_human_init_trans_in_first_cam).reshape(1, -1).to(device)

        first_cam_human_params[missing_human_name] = missing_human_params_in_other_cam

    # Visualize the first cam human parameters with the camera poses
    # decode the human parameters to 3D vertices and visualize
    if get_vertices:
        first_cam_human_vertices = {}
        for human_name, human_params in first_cam_human_params.items():
            body_pose = human_params['body_pose'].reshape(1, -1).to(device)
            global_orient = human_params['global_orient'].reshape(1, -1).to(device)
            betas = human_params['betas'].reshape(1, -1).to(device)
            left_hand_pose = human_params['left_hand_pose'].reshape(1, -1).to(device)
            right_hand_pose = human_params['right_hand_pose'].reshape(1, -1).to(device)
            transl = human_params['transl'].reshape(1, -1).to(device)

            smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=transl)

            vertices = smplx_output.vertices[0].detach().cpu().numpy()
            joints = smplx_output.joints[0].detach().cpu().numpy()
            vertices = vertices - joints[0:1] + human_params['root_transl'].cpu().numpy()
            first_cam_human_vertices[human_name] = vertices
            # visualize_cameras_and_human(cam_poses, human_vertices=first_cam_human_vertices, smplx_faces=smplx_layer.faces)
    else:
        first_cam_human_vertices = None

    optim_target_dict = {} # human_name: str -> Dict[param_name: str -> nn.Parameter]
    for human_name, human_params in first_cam_human_params.items():
        optim_target_dict[human_name] = {}
        
        # Convert human parameters to nn.Parameters for optimization
        optim_target_dict[human_name]['body_pose'] = nn.Parameter(human_params['body_pose'].float().to(device))  # (1, 63)
        optim_target_dict[human_name]['global_orient'] = nn.Parameter(human_params['global_orient'].float().to(device))  # (1, 3) 
        optim_target_dict[human_name]['betas'] = nn.Parameter(human_params['betas'].float().to(device))  # (1, 10)
        optim_target_dict[human_name]['left_hand_pose'] = nn.Parameter(human_params['left_hand_pose'].float().to(device))  # (1, 45)
        optim_target_dict[human_name]['right_hand_pose'] = nn.Parameter(human_params['right_hand_pose'].float().to(device))  # (1, 45)
        optim_target_dict[human_name]['root_transl'] = nn.Parameter(human_params['root_transl'].float().to(device)) # (1, 3)

        # TEMP
        # make relative_rotvec, shape, expression not require grad
        optim_target_dict[human_name]['body_pose'].requires_grad = False
        # optim_target_dict[human_name]['global_orient'].requires_grad = False
        optim_target_dict[human_name]['betas'].requires_grad = False
        optim_target_dict[human_name]['left_hand_pose'].requires_grad = False
        optim_target_dict[human_name]['right_hand_pose'].requires_grad = False

    return optim_target_dict, cam_poses, first_cam_human_vertices


def get_stage_optimizer(human_params, scene_params, residual_scene_scale, stage: int, lr: float = 0.01, device: str = 'cuda'):
    # 1st stage; optimize the scene scale, human root translation, shape (beta), and global orientation parameters
    # 2nd stage; optimize the dust3r scene parameters +  human root translation, shape (beta), and global orientation
    # 3rd stage; 2nd stage + human local poses
    # human param names: ['root_transl', 'betas', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose']

    # Set which parameters to optimize for each stage
    if stage == 1:
        optimizing_param_names = ['root_transl', 'betas']
        optimizing_params = []
        for human_name, optim_target_dict in human_params.items():
            for param_name in optim_target_dict.keys():
                if param_name in optimizing_param_names:
                    optim_target_dict[param_name].requires_grad = True    
                    optimizing_params.append(optim_target_dict[param_name])
                else:
                    optim_target_dict[param_name].requires_grad = False
        optimizing_params.append(residual_scene_scale)

    elif stage == 2:
        optimizing_human_param_names = ['root_transl', 'betas']
        human_params_to_optimize = []
        for human_name, optim_target_dict in human_params.items():
            for param_name in optim_target_dict.keys():
                if param_name in optimizing_human_param_names:
                    optim_target_dict[param_name].requires_grad = True
                    human_params_to_optimize.append(optim_target_dict[param_name])
                else:
                    optim_target_dict[param_name].requires_grad = False
        optimizing_params = scene_params + human_params_to_optimize

    elif stage == 3:
        optimizing_human_param_names = ['root_transl', 'betas', 'global_orient', 'body_pose']
        human_params_to_optimize = []
        for human_name, optim_target_dict in human_params.items():
            for param_name in optim_target_dict.keys():
                if param_name in optimizing_human_param_names:
                    optim_target_dict[param_name].requires_grad = True
                    human_params_to_optimize.append(optim_target_dict[param_name])
                else:
                    optim_target_dict[param_name].requires_grad = False
        optimizing_params = scene_params + human_params_to_optimize

    # Initialize LBFGS optimizer
    optimizer = torch.optim.LBFGS(
        optimizing_params,
        lr=lr,
        max_iter=4,
        # max_eval=25,
        tolerance_grad=1e-7,
        tolerance_change=1e-9,
        history_size=10,
        line_search_fn='strong_wolfe'
    )
    
    return optimizer

def vis_decode_human_params_and_cameras(world_multiple_human_3d_annot, cam_poses, smpl_layer, world_colmap_pointcloud_xyz, world_colmap_pointcloud_rgb, device='cuda'):
    # human_params: Dict[human_name -> Dict[param_name -> torch.Tensor]]
    # cam_poses: Dict[cam_name -> np.ndarray] (4,4)

    world_human_vertices = {}
    for human_name, human_params in world_multiple_human_3d_annot.items():
        body_pose = torch.from_numpy(human_params['body_pose']).reshape(1, -1).to(device).float()
        global_orient = torch.from_numpy(human_params['global_orient']).reshape(1, -1).to(device).float()
        betas = torch.from_numpy(human_params['betas']).reshape(1, -1).to(device).float()
    
        smpl_output = smpl_layer(betas=betas,
                                body_pose=body_pose,
                                global_orient=global_orient,
                                pose2rot=True,
                            )
        root_transl = human_params['root_transl'] # np.ndarray (1, 3)
        vertices = smpl_output.vertices.detach().squeeze(0).cpu().numpy()
        joints = smpl_output.joints.detach().squeeze(0).cpu().numpy()
        vertices = vertices - joints[0:1:, ] + root_transl
        world_human_vertices[human_name] = vertices

    visualize_cameras_and_human(cam_poses, human_vertices=world_human_vertices, smplx_faces=smpl_layer.faces, world_colmap_pointcloud_xyz=world_colmap_pointcloud_xyz, world_colmap_pointcloud_rgb=world_colmap_pointcloud_rgb)


def show_optimization_results(world_env, human_params, smplx_layer):
    smplx_vertices_dict = {}
    for human_name, optim_target_dict in human_params.items():
        # extract data from the optim_target_dict
        body_pose = optim_target_dict['body_pose'].reshape(1, -1)
        betas = optim_target_dict['betas'].reshape(1, -1)
        global_orient = optim_target_dict['global_orient'].reshape(1, -1)
        left_hand_pose = optim_target_dict['left_hand_pose'].reshape(1, -1)
        right_hand_pose = optim_target_dict['right_hand_pose'].reshape(1, -1)

        # decode the smpl mesh and joints
        smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)

        # Add root translation to the joints
        root_transl = optim_target_dict['root_transl'].reshape(1, 1, -1)
        smplx_vertices = smplx_output.vertices
        smplx_j3d = smplx_output.joints # (1, J, 3), joints in the world coordinate from the world mesh decoded by the optimizing parameters
        smplx_vertices = smplx_vertices - smplx_j3d[:, 0:1, :] + root_transl
        smplx_j3d = smplx_j3d - smplx_j3d[:, 0:1, :] + root_transl # !ALWAYS! Fuck the params['transl']
        smplx_vertices_dict[human_name] = smplx_vertices[0].detach().cpu().numpy()
    try:
        show_env_human_in_viser(world_env=world_env, world_scale_factor=1., smplx_vertices_dict=smplx_vertices_dict, smplx_faces=smplx_layer.faces)
    except:
        import pdb; pdb.set_trace()
    
def convert_human_params_to_numpy(human_params):
    # convert human_params to numpy arrays and save to new dictionary
    human_params_np = {}
    for human_name, optim_target_dict in human_params.items():
        human_params_np[human_name] = {}
        for param_name in optim_target_dict.keys():
            human_params_np[human_name][param_name] = optim_target_dict[param_name].reshape(1, -1).detach().cpu().numpy()

    return human_params_np

def main(output_dir: str = './outputs/egohumans/', sel_big_seqs: List = [], sel_small_seq_range: List[int] = [], optimize_human: bool = True, dust3r_raw_output_dir: str = './outputs/egohumans/dust3r_raw_outputs/dust3r_raw_outputs_random_sampled_views', dust3r_ga_output_dir: str = './outputs/egohumans/dust3r_ga_outputs_and_gt_cameras/dust3r_ga_outputs_and_gt_cameras_random_sampled_views', vitpose_hmr2_hamer_output_dir: str = '/scratch/one_month/2024_10/lmueller/egohuman/camera_ready', identified_vitpose_hmr2_hamer_output_dir: str = '/scratch/partial_datasets/egoexo/hongsuk/egohumans/vitpose_hmr2_hamer_predictions', egohumans_data_root: str = './data/egohumans_data', vis: bool = False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    vis_output_path = osp.join(output_dir, 'vis')
    Path(vis_output_path).mkdir(parents=True, exist_ok=True)

    # Parameters I am tuning
    human_loss_weight = 5.0
    stage2_start_idx_percentage = 0.2
    stage3_start_idx_percentage = 0.7
    niter = 600
    lr = 0.01
    dist_tol = 0.3
    scale_increasing_factor = 1.2
    # # TEMP
    # identified_vitpose_hmr2_hamer_output_dir = None

    # EgoHumans data
    # Fix batch size to 1 for now   
    selected_big_seq_list = sel_big_seqs #['03_fencing'] # #['07_tennis'] #  # #['01_tagging', '02_lego, 05_volleyball', '04_basketball', '03_fencing'] # ##[, , ''] 
    selected_small_seq_start_and_end_idx_tuple = None if len(sel_small_seq_range) == 0 else sel_small_seq_range # ex) [0, 10]
    cam_names = None #sorted(['cam01', 'cam02', 'cam03', 'cam04'])
    # num_of_cams = 3
    num_of_cams = 4
    subsample_rate = 100 
    dust3r_raw_output_dir = osp.join(dust3r_raw_output_dir, f'num_of_cams{num_of_cams}')
    dust3r_ga_output_dir = osp.join(dust3r_ga_output_dir, f'num_of_cams{num_of_cams}')
    optim_output_dir = osp.join(output_dir, 'optim_outputs', 'optim_outputs_trial1', f'num_of_cams{num_of_cams}')
    Path(optim_output_dir).mkdir(parents=True, exist_ok=True)
    dataset, dataloader = create_dataloader(egohumans_data_root, optimize_human=optimize_human, dust3r_raw_output_dir=dust3r_raw_output_dir, dust3r_ga_output_dir=dust3r_ga_output_dir, vitpose_hmr2_hamer_output_dir=vitpose_hmr2_hamer_output_dir, identified_vitpose_hmr2_hamer_output_dir=identified_vitpose_hmr2_hamer_output_dir, batch_size=1, split='test', subsample_rate=subsample_rate, cam_names=cam_names, num_of_cams=num_of_cams, selected_big_seq_list=selected_big_seq_list, selected_small_seq_start_and_end_idx_tuple=selected_small_seq_start_and_end_idx_tuple)

    # Dust3r Config for the global alignment
    mode = GlobalAlignerMode.PointCloudOptimizer if num_of_cams > 2 else GlobalAlignerMode.PairViewer
    device = 'cuda'
    silent = False
    schedule = 'linear'
    lr_base = lr
    lr_min = 0.0001
    init = 'known_params_hongsuk'
    niter_PnP = 10
    min_conf_thr_for_pnp = 3
    norm_pw_scale = False

    # Human related Config
    shape_prior_weight = 1.0
    human_lr = lr * 1.0 # not really used; to use modify the adjust_lr function; define different learning rate for the human parameters
    # set smplx layers
    smplx_layer_dict = {
        1: smplx.create(model_path = '/home/hongsuk/projects/egoexo/essentials/body_models', model_type = 'smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, use_face_contour = True, num_betas = 10, batch_size = 1).to(device),
        2: smplx.create(model_path = '/home/hongsuk/projects/egoexo/essentials/body_models', model_type = 'smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, use_face_contour = True, num_betas = 10, batch_size = 2).to(device),
        3: smplx.create(model_path = '/home/hongsuk/projects/egoexo/essentials/body_models', model_type = 'smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, use_face_contour = True, num_betas = 10, batch_size = 3).to(device),
        4: smplx.create(model_path = '/home/hongsuk/projects/egoexo/essentials/body_models', model_type = 'smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, use_face_contour = True, num_betas = 10, batch_size = 4).to(device),
    }
    smplx_layer = smplx_layer_dict[1]
    smpl_layer = smplx.create('./models', "smpl") # for GT
    smpl_layer = smpl_layer.to(device).float()

    # Logistics 
    save_2d_pose_vis = 20 
    scene_loss_timer = Timer()
    human_loss_timer = Timer()
    gradient_timer = Timer()

    total_output = {}
    total_scene_num = len(dataset)
    print(f"Running global alignment for {total_scene_num} scenes")
    for i in tqdm.tqdm(range(total_scene_num), total=total_scene_num):
        sample = dataset.get_single_item(i)

        world_multiple_human_3d_annot = sample['world_multiple_human_3d_annot']
        world_gt_cameras = sample['multiview_cameras']

        # Visualize the groundtruth human parameters and cameras
        world_cam_poses = {}
        for cam_name in world_gt_cameras.keys():
            cam2world = world_gt_cameras[cam_name]['cam2world_4by4']
            world_cam_poses[cam_name] = cam2world
        world_colmap_pointcloud_xyz = sample['world_colmap_pointcloud_xyz']
        world_colmap_pointcloud_rgb = sample['world_colmap_pointcloud_rgb']
        if False and vis:
            try:
                vis_decode_human_params_and_cameras(world_multiple_human_3d_annot, world_cam_poses, smpl_layer, world_colmap_pointcloud_xyz, world_colmap_pointcloud_rgb, device)
            except:
                import pdb; pdb.set_trace()

        # TEMPORARY Sanity check; due to the reid issue because of the noisy 2D groundtruth annotation, some views don't have any human detections, which doesn't make sense
        # Skip those samples
        sample_cam_names = list(sample['multiview_multiple_human_cam_pred'].keys())
        sanity_check_skip = False
        for cam_name in sample['multiview_multiple_human_cam_pred'].keys():
            if len(sample['multiview_multiple_human_cam_pred'][cam_name].keys()) == 0:
                sanity_check_skip = True
                break
        if sanity_check_skip:
            print("\nSkipping this sample due to no human detections in at least one camera view...")
            print(f"Skipping sample: {sample['sequence']}_{sample['frame']}_{''.join(sample_cam_names)}...")
            continue

        """ Get dust3r network output and global alignment results """
        dust3r_network_output = sample['dust3r_network_output']

        # load the precomputed 3D points, camera poses, and intrinsics from Dust3r GA output
        dust3r_ga_output = sample['dust3r_ga_output']
        pts3d, im_focals, im_poses, cam_names = get_resume_info(dust3r_ga_output, device)
        assert sample_cam_names == cam_names, "Camera names do not match"
        
        # intrinsics for human translation initialization
        init_focal_length = im_focals[0] #scene.get_intrinsics()[0][0].detach().cpu().numpy()
        init_princpt = [256., 144.] #scene.get_intrinsics()[0][:2, 2].detach().cpu().numpy()

        """ Initialize the human parameters """
        print("Initializing human parameters")
        multiview_multiple_human_cam_pred = sample['multiview_multiple_human_cam_pred'] # Dict[camera_name -> Dict[human_name -> 'pose2d', 'bbox', 'params' Dicts]]
        # pose2d and bbox are used for optimization target
        # params are used for initialization; 3D parameters.

        print("Initializing 2D pose and bbox")
        multiview_affine_transforms = sample['multiview_affine_transforms'] # Dict[camera_name -> np.ndarray]
        multiview_images = sample['multiview_images'] # Dict[camera_name -> Dict] # for visualization
        multiview_multiperson_poses2d = defaultdict(dict)
        multiview_multiperson_bboxes = defaultdict(dict)
        # make a dict of human_name -> Dict[cam_name -> (J, 3)]
        # make a dict of human_name -> Dict[cam_name -> (5)] for bboxes; xywh confidence
        for cam_name in multiview_multiple_human_cam_pred.keys():
            for human_name in multiview_multiple_human_cam_pred[cam_name].keys():
                pose2d = multiview_multiple_human_cam_pred[cam_name][human_name]['pose2d']
                pose2d[:, 2] = 1
                bbox = multiview_multiple_human_cam_pred[cam_name][human_name]['bbox']
                # just set confidence to 1; it seems to be corrupted in the data
                bbox[4] = 1

                # affine transform the 2D joints
                multiview_affine_transform = multiview_affine_transforms[cam_name]
                # make it to 3by3 matrix
                multiview_affine_transform = np.concatenate((multiview_affine_transform, np.array([[0, 0, 1]])), axis=0)
                # get inverse; original image space to dust3r image space
                multiview_affine_transform = np.linalg.inv(multiview_affine_transform)
                pose2d = multiview_affine_transform @ pose2d.T
                pose2d = pose2d.T
                pose2d[:, 2] = multiview_multiple_human_cam_pred[cam_name][human_name]['pose2d'][:, 2] # confidence

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
                    # draw the human name
                    img = cv2.putText(img, human_name, (int(bbox[0]), int(bbox[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.imwrite(osp.join(vis_output_path, f'target_{sample["sequence"]}_{sample["frame"]}_{cam_name}_{human_name}_2d_keypoints_bbox.png'), img[..., ::-1])

                multiview_multiperson_poses2d[human_name][cam_name] = torch.from_numpy(pose2d).to(device)
                multiview_multiperson_bboxes[human_name][cam_name] = torch.from_numpy(bbox).to(device)

        print("Initializing human parameters to optimize")
        human_params_to_optimize = []
        human_params_names_to_optimize = []

        human_params, human_inited_cam_poses, first_cam_human_vertices = \
            init_human_params(smplx_layer_dict[1], multiview_multiple_human_cam_pred, multiview_multiperson_poses2d, init_focal_length, init_princpt, device, get_vertices=vis) # dict of human parameters
        for human_name, optim_target_dict in human_params.items():
            for param_name in optim_target_dict.keys():
                if optim_target_dict[param_name].requires_grad:
                    human_params_to_optimize.append(optim_target_dict[param_name])
                    human_params_names_to_optimize.append(f'{human_name}_{param_name}')
        print(f"Optimizing {len(human_params_names_to_optimize)} parameters of humans: {human_params_names_to_optimize}")

        """ Initialize the scene parameters """
        # Initialize the scale factor between the dust3r cameras and the human_inited_cam_poses
        # Perform Procrustes alignment
        human_inited_cam_locations = []
        dust3r_cam_locations = []
        for cam_name in human_inited_cam_poses.keys():
            human_inited_cam_locations.append(human_inited_cam_poses[cam_name][:3, 3])
            dust3r_cam_locations.append(im_poses[cam_names.index(cam_name)][:3, 3].cpu().numpy())
        human_inited_cam_locations = np.array(human_inited_cam_locations)
        dust3r_cam_locations = np.array(dust3r_cam_locations)

        try:
            if len(human_inited_cam_locations) > 2:
                scene_scale, _, _ = procrustes_align(human_inited_cam_locations, dust3r_cam_locations)
            elif len(human_inited_cam_locations) == 2:
                # get the ratio between the two distances
                dist_ratio = np.linalg.norm(human_inited_cam_locations[0] - human_inited_cam_locations[1]) / np.linalg.norm(dust3r_cam_locations[0] - dust3r_cam_locations[1])
                scene_scale = dist_ratio
            else:
                print("Not enough camera locations to perform Procrustes alignment or distance ratio calculation")
                scene_scale = 80.0
            # Scale little bit larget to ensure humans are inside the views
            scene_scale = scene_scale * scale_increasing_factor
        except:
            print("Error in Procrustes alignment or distance ratio calculation due to zero division...")
            print(f"Skipping this sample {sample['sequence']}_{sample['frame']}_{''.join(cam_names)}...")
            continue

        print(f"Dust3r to Human scale factor: {scene_scale}")
        # do the optimization again with scaled 3D points and camera poses
        pts3d_scaled = [p * scene_scale for p in pts3d]
        pts3d = pts3d_scaled
        im_poses[:, :3, 3] = im_poses[:, :3, 3] * scene_scale

        # define the scene class that will be optimized
        scene = global_aligner(dust3r_network_output, device=device, mode=mode, verbose=not silent, has_human_cue=False)
        scene.norm_pw_scale = norm_pw_scale

        # initialize the scene parameters with the known poses or point clouds
        if num_of_cams > 2:
            if init == 'mst':
                scene.init_default_mst(niter_PnP=niter_PnP, min_conf_thr=min_conf_thr_for_pnp)
                print("Default MST init")
            elif init == 'known_params_hongsuk':
                scene.init_from_known_params_hongsuk(im_focals=im_focals, im_poses=im_poses, pts3d=pts3d, niter_PnP=niter_PnP, min_conf_thr=min_conf_thr_for_pnp)
                print("Known params init")

        scene_params = [p for p in scene.parameters() if p.requires_grad]

        # Visualize the initilization of 3D human and 3D world
        if False and first_cam_human_vertices is not None:
            world_env = parse_to_save_data(scene, cam_names)
            try:
                show_env_human_in_viser(world_env=world_env, world_scale_factor=1., smplx_vertices_dict=first_cam_human_vertices, smplx_faces=smplx_layer.faces)
            except:
                import pdb; pdb.set_trace()

        # Compute the sum of distances between camera centers; im_poses[:, :3, 3]
        # Get camera centers from im_poses
        init_cam_centers = scene.get_im_poses()[:, :3, 3]  # (N, 3)
        
        # Compute distances between all pairs of cameras
        init_cam_center_dist_total = 0
        init_cam_center_dist_totalpairs_dist = []
        for i in range(len(init_cam_centers)):
            for j in range(i+1, len(init_cam_centers)):
                dist = torch.norm(init_cam_centers[i] - init_cam_centers[j]).detach()
                # init_cam_center_dist_total += dist
                init_cam_center_dist_totalpairs_dist.append(dist)
        init_cam_center_dist_total = torch.stack(init_cam_center_dist_totalpairs_dist)
        # recover the original initial scale for regularization
        init_cam_center_dist_total = init_cam_center_dist_total / scale_increasing_factor

        # init_cam_center_dist_total = torch.sum(torch.stack(pairs_dist))
        # init_cam_center_dist_total = init_cam_center_dist_total.detach()
        # print(f"Initial sum of distances between camera centers: {init_cam_center_dist_total:.3f}")
        # set scene scale tolerance
        # scene_scale_tol = 0.15
        # print(f"Scene scale tolerance: {scene_scale_tol:.3f}")
        print("Distance tolerance for camera centers: ", dist_tol)

        # 1st stage; stage 1 is from 0% to 30%
        stage1_iter = list(range(0, int(niter * stage2_start_idx_percentage)))
        # 2nd stage; stage 2 is from 30% to 60%
        stage2_iter = list(range(int(niter * stage2_start_idx_percentage), int(niter * stage3_start_idx_percentage)))
        # 3rd stage; stage 3 is from 60% to 100%
        stage3_iter = list(range(int(niter * stage3_start_idx_percentage), niter))

        print(">>> Set the scene scale as a parameter to optimize")
        residual_scene_scale = nn.Parameter(torch.tensor(1., requires_grad=True).to(device))

        # Given the number of iterations, run the optimizer while forwarding the scene with the current parameters to get the loss
        with tqdm.tqdm(total=niter) as bar:
            while bar.n < bar.total:
                # Set optimizer
                if bar.n == stage1_iter[0]:
                    optimizer = get_stage_optimizer(human_params, scene_params, residual_scene_scale, 1, lr)
                    print("\n1st stage optimization starts at ", bar.n)
                elif bar.n == stage2_iter[0]:
                    optimizer = get_stage_optimizer(human_params, scene_params, residual_scene_scale, 2, lr)
                    print("\n2nd stage optimization starts at ", bar.n)
                    # Reinitialize the scene
                    # TEMP
                    print("Residual scene scale: ", residual_scene_scale.item())
                    scene_intrinsics = scene.get_intrinsics().detach().cpu().numpy()
                    im_focals = [intrinsic[0,0] * residual_scene_scale.item() for intrinsic in scene_intrinsics]
                    im_poses = scene.get_im_poses().detach()
                    im_poses[:, :3, 3] = im_poses[:, :3, 3] * residual_scene_scale.item()
                    pts3d = scene.get_pts3d()
                    pts3d_scaled = [p * residual_scene_scale.item() for p in pts3d]
                    scene.init_from_known_params_hongsuk(im_focals=im_focals, im_poses=im_poses, pts3d=pts3d_scaled, niter_PnP=niter_PnP, min_conf_thr=min_conf_thr_for_pnp)
                    print("Known params init")

                    if vis:
                        # Visualize the initilization of 3D human and 3D world
                        world_env = parse_to_save_data(scene, cam_names)
                        show_optimization_results(world_env, human_params, smplx_layer_dict[1])

                elif bar.n == stage3_iter[0]:
                    optimizer = get_stage_optimizer(human_params, scene_params, residual_scene_scale, 3, lr)
                    print("\n3rd stage optimization starts at ", bar.n)

                    if vis:
                        # Visualize the initilization of 3D human and 3D world
                        world_env = parse_to_save_data(scene, cam_names)
                        show_optimization_results(world_env, human_params, smplx_layer_dict[1])

                lr = adjust_lr(bar.n, niter, lr_base, lr_min, optimizer, schedule)

                # Define closure for LBFGS
                def closure():
                    optimizer.zero_grad()
                    
                    # Get extrinsics and intrinsics from the scene
                    multiview_cam2world_4by4 = scene.get_im_poses().detach()
                    
                    # if bar.n in stage1_iter:
                    #     # Scale camera translations
                    #     multiview_cam2world_3by4 = torch.cat([
                    #         multiview_cam2world_4by4[:, :3, :3],
                    #         (multiview_cam2world_4by4[:, :3, 3] * residual_scene_scale).unsqueeze(-1),
                    #     ], dim=2)
                    #     multiview_cam2world_4by4 = torch.cat([
                    #         multiview_cam2world_3by4,
                    #         multiview_cam2world_4by4[:, 3:4, :]
                    #     ], dim=1)
                    
                    multiview_world2cam_4by4 = torch.inverse(multiview_cam2world_4by4)
                    multiview_intrinsics = scene.get_intrinsics().detach()

                    # Initialize losses dictionary
                    losses = {}

                    # Get human loss
                    human_loss_timer.tic()
                    losses['human_loss'], projected_joints = get_human_loss(
                        smplx_layer_dict, human_params, cam_names, 
                        multiview_world2cam_4by4, multiview_intrinsics, 
                        multiview_multiperson_poses2d, multiview_multiperson_bboxes, 
                        shape_prior_weight, device
                    )
                    losses['human_loss'] = human_loss_weight * losses['human_loss']
                    human_loss_timer.toc()

                    # Get scene loss
                    if bar.n in stage2_iter or bar.n in stage3_iter:
                        scene_loss_timer.tic()
                        losses['scene_loss'] = scene.dust3r_loss()
                        scene_loss_timer.toc()

                    # Compute total loss
                    total_loss = sum(losses.values())
                    
                    # Compute gradients
                    gradient_timer.tic()
                    total_loss.backward()
                    gradient_timer.toc()

                    # Update progress bar
                    loss_str = f'{lr=:g} '
                    loss_str += ' '.join([f'{k}={v:g}' for k, v in losses.items()])
                    loss_str += f' total_loss={total_loss:g}'
                    bar.set_postfix_str(loss_str)
                    
                    # Visualize if needed
                    if vis and bar.n % save_2d_pose_vis == 0:
                        for cam_name, human_joints in projected_joints.items():
                            img = scene.imgs[cam_names.index(cam_name)].copy() * 255.
                            img = img.astype(np.uint8)
                            for human_name, joints in human_joints.items():
                                img = cv2.putText(img, human_name, (int(joints[0, 0]), int(joints[0, 1])), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                for idx, joint in enumerate(joints):
                                    img = cv2.circle(img, (int(joint[0]), int(joint[1])), 1, (0, 255, 0), -1)
                            cv2.imwrite(osp.join(vis_output_path, 
                                    f'{sample["sequence"]}_{sample["frame"]}_{cam_name}_{bar.n}.png'), 
                                    img[:, :, ::-1])
                    

                    return total_loss

            # Step the optimizer
            optimizer.step(closure)
            bar.update()
            print(f"Time taken: human_loss={human_loss_timer.total_time:g}s, scene_loss={scene_loss_timer.total_time:g}s, backward={gradient_timer.total_time:g}s")

            # Print final loss value
            final_loss = closure()
            print(f"Final total loss: {final_loss.item():.6f}")
            # print("Final losses:", ' '.join([f'{k}={v.item():g}' for k, v in losses.items()]))

        # Save output
        output_name = f"{sample['sequence']}_{sample['frame']}_{''.join(cam_names)}"
        total_output = {}
        # total_output['gt_cameras'] = sample['multiview_cameras']
        total_output['world_gt_cameras'] = world_gt_cameras
        total_output['world_multiple_human_3d_annot'] = world_multiple_human_3d_annot
        total_output['dust3r_ga'] = parse_to_save_data(scene, cam_names)
        # convert human_params to numpy arrays and save to new dictionary
        # human_params_np = {}
        # for human_name, optim_target_dict in human_params.items():
        #     human_params_np[human_name] = {}
        #     for param_name in optim_target_dict.keys():
        #         human_params_np[human_name][param_name] = optim_target_dict[param_name].reshape(1, -1).detach().cpu().numpy()
        # total_output['human_params'] = human_params_np
        total_output['human_params'] = convert_human_params_to_numpy(human_params)
        print("Saving to ", osp.join(optim_output_dir, f'{output_name}.pkl'))
        with open(osp.join(optim_output_dir, f'{output_name}.pkl'), 'wb') as f:
            pickle.dump(total_output, f)
        # 
        
        if vis:
            show_optimization_results(total_output['dust3r_ga'], human_params, smplx_layer_dict[1])
            # smplx_vertices_dict = {}
            # for human_name, optim_target_dict in human_params.items():
            #     # extract data from the optim_target_dict
            #     body_pose = optim_target_dict['body_pose'].reshape(1, -1)
            #     betas = optim_target_dict['betas'].reshape(1, -1)
            #     global_orient = optim_target_dict['global_orient'].reshape(1, -1)
            #     left_hand_pose = optim_target_dict['left_hand_pose'].reshape(1, -1)
            #     right_hand_pose = optim_target_dict['right_hand_pose'].reshape(1, -1)

            #     # decode the smpl mesh and joints
            #     smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose)

            #     # Add root translation to the joints
            #     root_transl = optim_target_dict['root_transl'].reshape(1, 1, -1)
            #     smplx_vertices = smplx_output.vertices
            #     smplx_j3d = smplx_output.joints # (1, J, 3), joints in the world coordinate from the world mesh decoded by the optimizing parameters
            #     smplx_vertices = smplx_vertices - smplx_j3d[:, 0:1, :] + root_transl
            #     smplx_j3d = smplx_j3d - smplx_j3d[:, 0:1, :] + root_transl # !ALWAYS! Fuck the params['transl']
            #     smplx_vertices_dict[human_name] = smplx_vertices[0].detach().cpu().numpy()
            # try:
            #     show_env_human_in_viser(world_env=total_output['dust3r_ga'], world_scale_factor=1., smplx_vertices_dict=smplx_vertices_dict, smplx_faces=smplx_layer.faces)
            # except:
            #     import pdb; pdb.set_trace()

if __name__ == '__main__':
    tyro.cli(main)