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
import smplx

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

        # for visualization purpose
        for idx, sam_cam_idx in enumerate(sampled_cam_indices):
            projected_joints[cam_names[sam_cam_idx]][human_name] = multiview_smplx_j2d_coco_ordered[idx]

    return human_loss, projected_joints

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


def main(output_dir: str = './outputs/egohumans/', optimize_human: bool = True, dust3r_raw_output_dir: str = './outputs/egohumans/dust3r_raw_outputs', dust3r_ga_output_dir: str = './outputs/egohumans/dust3r_ga_outputs_and_gt_cameras', egohumans_data_root: str = './data/egohumans_data', vis: bool = False):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    vis_output_path = osp.join(output_dir, 'vis')
    Path(vis_output_path).mkdir(parents=True, exist_ok=True)

    # EgoHumans data
    # Fix batch size to 1 for now
    selected_big_seq_list = [] #['07_tennis'] #  # #['01_tagging', '02_lego, 05_volleyball', '04_basketball', '03_fencing'] # ##[, , ''] 
    cam_names = None #sorted(['cam01', 'cam02', 'cam03', 'cam04'])
    num_of_cams = 4
    dust3r_raw_output_dir = osp.join(dust3r_raw_output_dir, f'num_of_cams{num_of_cams}')
    dust3r_ga_output_dir = osp.join(dust3r_ga_output_dir, f'num_of_cams{num_of_cams}')
    optim_output_dir = osp.join(output_dir, 'naive_optim_outputs', f'num_of_cams{num_of_cams}')
    Path(optim_output_dir).mkdir(parents=True, exist_ok=True)
    dataset, dataloader = create_dataloader(egohumans_data_root, optimize_human=optimize_human, dust3r_raw_output_dir=dust3r_raw_output_dir, dust3r_ga_output_dir=dust3r_ga_output_dir, batch_size=1, split='test', subsample_rate=10, cam_names=cam_names, num_of_cams=num_of_cams, selected_big_seq_list=selected_big_seq_list)

    # Dust3r Config for the global alignment
    mode = GlobalAlignerMode.PointCloudOptimizer if num_of_cams > 2 else GlobalAlignerMode.PairViewer
    device = 'cuda'
    silent = False
    schedule = 'linear'
    niter = 1000 #500
    lr = 0.01
    lr_base = lr
    lr_min = 0.0001
    init = 'known_params_hongsuk'
    niter_PnP = 10
    min_conf_thr_for_pnp = 3
    norm_pw_scale = False


    # define different learning rate for the human parameters
    human_lr = lr * 1.0
    smplx_layer = smplx.create(
        model_path = '/home/hongsuk/projects/egoexo/essentials/body_models',
        model_type = 'smplx',
        gender = 'neutral',
        use_pca = False,
        num_pca_comps = 45,
        flat_hand_mean = True,
        use_face_contour = True,
        num_betas = 10,
        batch_size = 1,
    )
    smplx_layer = smplx_layer.to('cuda')
    save_2d_pose_vis = 20 

    total_output = {}
    total_scene_num = len(dataset)
    print(f"Running global alignment for {total_scene_num} scenes")
    for i in tqdm.tqdm(range(total_scene_num), total=total_scene_num):
        # TEMP; subsample
        if i % 100 != 0:
            continue

        sample = dataset.get_single_item(i)

        """ Get dust3r network output and global alignment results """
        dust3r_network_output = sample['dust3r_network_output']

        # load the precomputed 3D points, camera poses, and intrinsics from Dust3r GA output
        dust3r_ga_output = sample['dust3r_ga_output']
        pts3d, im_focals, im_poses, cam_names = get_resume_info(dust3r_ga_output, device)
        
        # intrinsics for human translation initialization
        init_focal_length = im_focals[0] #scene.get_intrinsics()[0][0, 0].detach().cpu().numpy()
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
            init_human_params(smplx_layer, multiview_multiple_human_cam_pred, multiview_multiperson_poses2d, init_focal_length, init_princpt, device, get_vertices=False) # dict of human parameters
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

        scale, _, _ = procrustes_align(human_inited_cam_locations, dust3r_cam_locations)
        print(f"Dust3r to Human scale factor: {scale}")
        # do the optimization again with scaled 3D points and camera poses
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

        # Visualize the initilization of 3D human and 3D world
        if first_cam_human_vertices is not None:
            world_env = parse_to_save_data(scene, cam_names)
            show_env_human_in_viser(world_env=world_env, world_scale_factor=1., smplx_vertices_dict=first_cam_human_vertices, smplx_faces=smplx_layer.faces)

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
        # init_cam_center_dist_total = torch.sum(torch.stack(pairs_dist))
        # init_cam_center_dist_total = init_cam_center_dist_total.detach()
        # print(f"Initial sum of distances between camera centers: {init_cam_center_dist_total:.3f}")
        # set scene scale tolerance
        # scene_scale_tol = 0.15
        # print(f"Scene scale tolerance: {scene_scale_tol:.3f}")
        dist_tol = 0.15
        print("Distance tolerance for camera centers: ", dist_tol)


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

                # get the distances between camera centers
                cam_centers = multiview_cam2world_4by4[:, :3, 3]  # (N, 3)
                # cam_center_dist_total = 0
                cam_center_dist_totalpairs_dist = []
                for i in range(len(cam_centers)):
                    for j in range(i+1, len(cam_centers)):
                        dist = torch.norm(cam_centers[i] - cam_centers[j])
                        cam_center_dist_totalpairs_dist.append(dist)
                cam_center_dist_total = torch.stack(cam_center_dist_totalpairs_dist)
                # Compute the distance loss for each pair of cameras
                relative_dist_diff = torch.abs(cam_center_dist_total - init_cam_center_dist_total) / init_cam_center_dist_total
                cam_pose_dist_loss = torch.nn.functional.relu(relative_dist_diff - dist_tol) * 10.0  # Multiply by 10 to make the loss more significant when violated
                cam_pose_dist_loss = torch.mean(cam_pose_dist_loss)

                loss = scene_loss + 5 * human_loss + cam_pose_dist_loss

                loss.backward()
                optimizer.step()

                bar.set_postfix_str(f'{lr=:g} loss={loss:g}, scene_loss={scene_loss:g}, human_loss={human_loss:g}, cam_pose_dist_loss={cam_pose_dist_loss:g}')
                bar.update()

                if vis and bar.n % save_2d_pose_vis == 0:
                    for cam_name, human_joints in projected_joints.items():
                        img = scene.imgs[cam_names.index(cam_name)].copy() * 255.
                        img = img.astype(np.uint8)
                        for human_name, joints in human_joints.items():
                            # darw the human name
                            img = cv2.putText(img, human_name, (int(joints[0, 0]), int(joints[0, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            for idx, joint in enumerate(joints):
                                img = cv2.circle(img, (int(joint[0]), int(joint[1])), 1, (0, 255, 0), -1)
                                # draw the index
                                # img = cv2.putText(img, f"{idx}", (int(joint[0]), int(joint[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.imwrite(osp.join(vis_output_path, f'{sample["sequence"]}_{sample["frame"]}_{cam_name}_{bar.n}.png'), img[:, :, ::-1])
                
        print("final losses: ", scene_loss.item(), human_loss.item())

        # Save output
        output_name = f"{sample['sequence']}_{sample['frame']}_{''.join(cam_names)}"
        total_output = {}
        total_output['gt_cameras'] = sample['multiview_cameras']
        total_output['dust3r_ga'] = parse_to_save_data(scene, cam_names)
        # convert human_params to numpy arrays and save to new dictionary
        human_params_np = {}
        for human_name, optim_target_dict in human_params.items():
            human_params_np[human_name] = {}
            for param_name in optim_target_dict.keys():
                human_params_np[human_name][param_name] = optim_target_dict[param_name].reshape(1, -1).detach().cpu().numpy()
        total_output['human_params'] = human_params_np
        print("Saving to ", osp.join(optim_output_dir, f'{output_name}.pkl'))
        with open(osp.join(optim_output_dir, f'{output_name}.pkl'), 'wb') as f:
            pickle.dump(total_output, f)
        
        if vis:
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
                
            show_env_human_in_viser(world_env=total_output[output_name]['dust3r_ga'], world_scale_factor=1., smplx_vertices_dict=smplx_vertices_dict, smplx_faces=smplx_layer.faces)

if __name__ == '__main__':
    tyro.cli(main)