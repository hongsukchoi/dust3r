import os
import os.path as osp
import glob
import pickle
import tyro
import viser
import viser.transforms as vtf
import torch
import numpy as np
import smplx
import cv2
import copy 
import tqdm
import time
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from collections import defaultdict

from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.utils.device import to_numpy
from dust3r.optim_factory import adjust_learning_rate_by_lr
from dust3r.cloud_opt.commons import cosine_schedule, linear_schedule


from hongsuk_vis_viser_env_only import show_env_in_viser
from hongsuk_vis_viser_env_human import show_env_human_in_viser
from hongsuk_joint_names import COCO_WHOLEBODY_KEYPOINTS, ORIGINAL_SMPLX_JOINT_NAMES, COCO_MAIN_BODY_SKELETON, VITPOSEPLUS_KEYPOINTS

coco_main_body_end_joint_idx = COCO_WHOLEBODY_KEYPOINTS.index('right_heel') 
coco_main_body_joint_idx = list(range(coco_main_body_end_joint_idx + 1))
coco_main_body_joint_names = COCO_WHOLEBODY_KEYPOINTS[:coco_main_body_end_joint_idx + 1]
smplx_main_body_joint_idx = [ORIGINAL_SMPLX_JOINT_NAMES.index(joint_name) for joint_name in coco_main_body_joint_names] 
# Create mapping from VITPOSEPLUS_KEYPOINTS to COCO_WHOLEBODY_KEYPOINTS
vitposeplus_to_coco_mapper = [] # ex) vitposeplus_2d_keypoints[vitposeplus_to_coco_mapper] = coco_wholebody_2d_keypoints
for coco_idx, coco_joint_name in enumerate(COCO_WHOLEBODY_KEYPOINTS):
    # Direct name matches
    if coco_joint_name in VITPOSEPLUS_KEYPOINTS:
        vitposeplus_to_coco_mapper.append(VITPOSEPLUS_KEYPOINTS.index(coco_joint_name))
    else:
        print(f'Warning: {coco_joint_name} is not in VITPOSEPLUS_KEYPOINTS, adding None instead')
        vitposeplus_to_coco_mapper.append(None)
assert len(vitposeplus_to_coco_mapper) == len(COCO_WHOLEBODY_KEYPOINTS)
# Count non-None elements in vitposeplus_to_coco_mapper
non_none_count = sum(1 for x in vitposeplus_to_coco_mapper if x is not None)
assert non_none_count == 131, f"Expected 131 non-None elements in vitposeplus_to_coco_mapper, but got {non_none_count}"


def convert_human_params_to_numpy(human_params):
    # convert human_params to numpy arrays and save to new dictionary
    human_params_np = {}
    for human_name, optim_target_dict in human_params.items():
        human_params_np[human_name] = {}
        for param_name in optim_target_dict.keys():
            human_params_np[human_name][param_name] = optim_target_dict[param_name].reshape(1, -1).detach().cpu().numpy()

    return human_params_np


def draw_joints2d(joints2d, img_paths, output_dir, conf_threshold=0.2):
    # Draw joints2d on each image with index labels

    # Use single color for visualization
    joint_color = (0, 255, 0)  # Green color in BGR
    
    for cam_idx, img_path in enumerate(img_paths):
        # Read image
        img = cv2.imread(img_path)
        
        # Draw each joint
        for joint_idx, joint in enumerate(joints2d[cam_idx]):
            # Get x,y coordinates and confidence
            x, y, conf = joint
            
            # Skip if low confidence
            if conf < conf_threshold:
                continue
                
            # Convert to integer coordinates
            x = int(x)
            y = int(y)
            
            # Draw circle at joint location
            cv2.circle(img, (x,y), 3, joint_color, -1)
            
            # Draw joint index
            cv2.putText(img, str(joint_idx), (x+5,y+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, joint_color, 1)
            
        # Save annotated image
        output_path = osp.join(output_dir, f'joints2d_{cam_idx}.png')
        cv2.imwrite(output_path, img)

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

def show_optimization_results(world_env, human_params, smplx_layer):
    smplx_vertices_dict = {}
    for human_name, optim_target_dict in human_params.items():
        # extract data from the optim_target_dict
        body_pose = optim_target_dict['body_pose'].reshape(1, -1)
        betas = optim_target_dict['betas'].reshape(1, -1)
        global_orient = optim_target_dict['global_orient'].reshape(1, -1)
        left_hand_pose = optim_target_dict['left_hand_pose'].reshape(1, -1)
        right_hand_pose = optim_target_dict['right_hand_pose'].reshape(1, -1)
        transl = optim_target_dict['transl'].reshape(1, -1)
        # decode the smpl mesh and joints
        # smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=transl)

        smplx_output = smplx_layer(body_pose=body_pose.float(), betas=betas.float(), global_orient=global_orient.float(), left_hand_pose=left_hand_pose.float(), right_hand_pose=right_hand_pose.float(), transl=transl.float())

        smplx_vertices_dict[human_name] = smplx_output.vertices[0].detach().cpu().numpy()

    try:
        show_env_human_in_viser(world_env=world_env, world_scale_factor=1., smplx_vertices_dict=smplx_vertices_dict, smplx_faces=smplx_layer.faces)
    except:
        pass

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

# initialize human parameters to be optimized == nn.Parameters
def init_human_params(human_params, device):
    optim_target_dict = {} # human_name: str -> Dict[param_name: str -> nn.Parameter]
    for human_name, human_params in human_params.items():
        optim_target_dict[human_name] = {}
        
        # Convert human parameters to nn.Parameters for optimization
        optim_target_dict[human_name]['body_pose'] = nn.Parameter(human_params['body_pose'].float().to(device))  # (1, 63)
        optim_target_dict[human_name]['global_orient'] = nn.Parameter(human_params['global_orient'].float().to(device))  # (1, 3) 
        optim_target_dict[human_name]['betas'] = nn.Parameter(human_params['betas'].float().to(device))  # (1, 10)
        optim_target_dict[human_name]['left_hand_pose'] = nn.Parameter(human_params['left_hand_pose'].float().to(device))  # (1, 45)
        optim_target_dict[human_name]['right_hand_pose'] = nn.Parameter(human_params['right_hand_pose'].float().to(device))  # (1, 45)
        optim_target_dict[human_name]['transl'] = nn.Parameter(human_params['transl'].float().to(device)) # (1, 3)


    return optim_target_dict

def get_stage_optimizer(human_params, scene_params, residual_scene_scale, stage: int, lr: float = 0.01):
    # 1st stage; optimize the scene scale, human root translation, shape (beta), and global orientation parameters
    # 2nd stage; optimize the dust3r scene parameters +  human root translation, shape (beta), and global orientation
    # 3rd stage; 2nd stage + human local poses
    # human param names: ['transl', 'betas', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose']

    if stage == 1: # 1st
        optimizing_param_names = ['transl', 'betas'] # , 'global_orient'

        human_params_to_optimize = []
        human_params_names_to_optimize = []
        for human_name, optim_target_dict in human_params.items():
            for param_name in sorted(list(optim_target_dict.keys())):
                if param_name in optimizing_param_names:
                    optim_target_dict[param_name].requires_grad = True    
                    human_params_to_optimize.append(optim_target_dict[param_name])
                    human_params_names_to_optimize.append(f'{human_name}_{param_name}')
                else:
                    optim_target_dict[param_name].requires_grad = False

        optimizing_params = human_params_to_optimize + [residual_scene_scale]

    elif stage == 2: # 2nd
        optimizing_human_param_names = ['transl', 'betas', 'global_orient']

        human_params_to_optimize = []
        human_params_names_to_optimize = []
        for human_name, optim_target_dict in human_params.items():
            for param_name in sorted(list(optim_target_dict.keys())):
                if param_name in optimizing_human_param_names:
                    optim_target_dict[param_name].requires_grad = True
                    human_params_to_optimize.append(optim_target_dict[param_name])
                    human_params_names_to_optimize.append(f'{human_name}_{param_name}')
                else:
                    optim_target_dict[param_name].requires_grad = False

        optimizing_params = scene_params + human_params_to_optimize  # TEMP
        # optimizing_params = human_params_to_optimize 

    elif stage == 3: # 3rd
        optimizing_human_param_names = ['transl', 'betas', 'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose']

        human_params_to_optimize = []
        human_params_names_to_optimize = []
        for human_name, optim_target_dict in human_params.items():
            for param_name in sorted(list(optim_target_dict.keys())):
                if param_name in optimizing_human_param_names:
                    optim_target_dict[param_name].requires_grad = True
                    human_params_to_optimize.append(optim_target_dict[param_name])
                    human_params_names_to_optimize.append(f'{human_name}_{param_name}')
                else:
                    optim_target_dict[param_name].requires_grad = False

        optimizing_params = scene_params + human_params_to_optimize
    # Print optimization parameters
    print(f"Optimizing {len(optimizing_params)} parameters:")
    print(f"- Human parameters ({len(human_params_names_to_optimize)}): {human_params_names_to_optimize}")
    if stage == 2 or stage == 3:
        print(f"- Scene parameters ({len(scene_params)})")
    if stage == 1:
        print(f"- Residual scene scale (1)")
    optimizer = torch.optim.Adam(optimizing_params, lr=lr, betas=(0.9, 0.9))
    return optimizer


def get_human_loss(smplx_layer_dict, num_of_humans_for_optimization, humans_optim_target_dict, cam_names, multiview_world2cam_4by4, multiview_intrinsics, multiview_multiperson_poses2d, multiview_multiperson_bboxes, shape_prior_weight=0, device='cuda'):
    # multiview_multiperson_poses2d: Dict[human_name -> Dict[cam_name -> (J, 3)]]
    # multiview_multiperson_bboxes: Dict[human_name -> Dict[cam_name -> (5)]]
    # multiview_world2cam_4by4: (N, 4, 4), multiview_intrinsics: (N, 3, 3)

    # save the 2D joints for visualization
    projected_joints = defaultdict(dict)

    # Collect all human parameters into batched tensors
    human_names = sorted(list(humans_optim_target_dict.keys()))
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
    transl = torch.cat([humans_optim_target_dict[name]['transl'].reshape(1, -1) for name in human_names], dim=0)

    # Forward pass through SMPL-X model for all humans at once
    smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=transl)

    # Add root translation to joints
    smplx_j3d = smplx_output.joints  # (B, J, 3)

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
        
        if human_idx >= num_of_humans_for_optimization:
            break

    # Add shape prior if requested
    if shape_prior_weight > 0:
        total_loss += shape_prior_weight * F.mse_loss(betas[:num_of_humans_for_optimization], torch.zeros_like(betas[:num_of_humans_for_optimization]))

    return total_loss, projected_joints

def get_bboxes_from_joints2d(joints2d: np.ndarray):
    # joints2d: (num_cams, 135, 3) where last dim is (x,y,conf)
    # return: (num_cams, 5) where last dim is (x1,y1,x2,y2,conf)
    
    num_cams = joints2d.shape[0]
    bboxes = np.zeros((num_cams, 5))
    
    for cam_idx in range(num_cams):
        # Get valid joints (confidence > 0)
        valid_joints = joints2d[cam_idx, :, :2][joints2d[cam_idx, :, 2] > 0]
        
        if len(valid_joints) > 0:
            # Get min/max x,y coordinates for bounding box
            x1 = np.min(valid_joints[:, 0])
            y1 = np.min(valid_joints[:, 1]) 
            x2 = np.max(valid_joints[:, 0])
            y2 = np.max(valid_joints[:, 1])
            
            # Average confidence of all joints
            conf = np.mean(joints2d[cam_idx, :, 2])
            
            bboxes[cam_idx] = [x1, y1, x2, y2, conf]
            
    return bboxes

def show_ground_truth_scene(ground_truth, scene_point_size=0.006, point_shape='circle'):
    # add the ground truth scene, joints, and camera
    # set viser
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+y")

    # add ground truth dust3r points
    gt_aria_points_xyz = ground_truth['gt_scene_xyz'].cpu().numpy()
    server.scene.add_point_cloud(
        f'gt-aria-scene',
        points=gt_aria_points_xyz,
        point_size=scene_point_size,
        point_shape=point_shape,
        colors=(0.5,0.5,0.5)
    )

    # add the ground truth cameras 
    gt_extrins = ground_truth['gt_cam_T'].cpu().numpy()
    gt_intrins = ground_truth['gt_cam_K'].cpu().numpy()
    for cam_idx, cam_pose in enumerate(gt_extrins):
        # add camera axes
        server.scene.add_batched_axes(
            f'gt-camera-{cam_idx}', 
            batched_wxyzs=np.array(
                # Convert Nx3x3 rotation matrices to Nx4 quaternions.
                [vtf.SO3.from_matrix(cam_pose[:3, :3]).wxyz]
            ),
            batched_positions=np.array([cam_pose[:3, 3]]),
            axes_length=1.0,
            axes_radius=0.01,
        )
        # add camera frustum
        fx = gt_intrins[cam_idx][0, 0]
        iw = gt_intrins[cam_idx][0, 2]
        ih = gt_intrins[cam_idx][1, 2]
        fov = 2 * np.arctan(iw / 2 * fx) * 180 / np.pi
        server.scene.add_camera_frustum(
            f'gt-camera-frustum-{cam_idx}',
            fov=fov,
            aspect= iw / ih,
            wxyz=vtf.SO3.from_matrix(cam_pose[:3, :3]).wxyz,
            position=cam_pose[:3, 3],
            color=[0, 1, 0]
        )
    # set break point for debugging
    import pdb; pdb.set_trace()
    print("Done visualizing")

def main(output_dir: str = './outputs/egoexo/nov11/sota_comparison_tria1', use_gt_focal: bool = False, vis: bool = False, dust3r_network_output_path: str = '/scratch/partial_datasets/egoexo/preprocess_20241110_camera_ready/takes/utokyo_cpr_2005_34_2/preprocessing/dust3r_world_env_2/000384/images/dust3r_network_output_pointmaps_images.pkl', vitpose_and_gt_path: str = '/scratch/partial_datasets/egoexo/egoexo4d_v2_mvopti/run_04/train/utokyo_cpr_2005_34_2/384/input_data.pkl'):
    # Pipeline
    # 1) dust3r_network_output from /scratch/partial_datasets/egoexo/preprocess_20241110_camera_ready/takes/utokyo_cpr_2005_34_2/preprocessing/dust3r_world_env_2/000384/images
    # 2) im_focals (K), im_poses (T) from /scratch/partial_datasets/egoexo/egoexo4d_v2_mvopti/run_04/train/utokyo_cpr_2005_34_2/384/results/prefit.pkl, ['cam']['K'], ['cam']['T']
    # 3) load human parameter from /scratch/partial_datasets/egoexo/egoexo4d_v2_mvopti/run_04/train/utokyo_cpr_2005_34_2/384/results/prefit.pkl, ['human']
    # 4) load ViTPose and byte track results from /scratch/partial_datasets/egoexo4d_v2/processed_20240718/takes/utokyo_cpr_2005_34_2/preprocessing/cam0*__vitpose.pkl
    # 5) Final optimization
    # dust3r_network_output_path = '/scratch/partial_datasets/egoexo/preprocess_20241110_camera_ready/takes/utokyo_cpr_2005_34_2/preprocessing/dust3r_world_env_2/000384/images/dust3r_network_output_pointmaps_images.pkl'
    # prefit_path = '/scratch/partial_datasets/egoexo/egoexo4d_v2_mvopti/run_04/train/utokyo_cpr_2005_34_2/384/results/prefit.pkl'
    # vitpose_and_gt_path = '/scratch/partial_datasets/egoexo/egoexo4d_v2_mvopti/run_04/train/utokyo_cpr_2005_34_2/384/input_data.pkl'
    sequence_name = dust3r_network_output_path.split('/')[6]  #'utokyo_cpr_2005_34_2'
    frame_idx = int(dust3r_network_output_path.split('/')[-3]) #384 # for testing

    # vitpose_path_pattern = '/scratch/partial_datasets/egoexo4d_v2/processed_20240718/takes/utokyo_cpr_2005_34_2/preprocessing/cam0*__vitpose.pkl'
    # bytetrack_path_pattern = '/scratch/partial_datasets/egoexo4d_v2/processed_20240718/takes/utokyo_cpr_2005_34_2/preprocessing/cam0*__bbox_with_score_used.pkl'
    # vitpose_paths = sorted(glob.glob(vitpose_path_pattern))
    # bytetrack_paths = sorted(glob.glob(bytetrack_path_pattern))
    img_paths = sorted(glob.glob(osp.join(osp.dirname(dust3r_network_output_path), '*.png')))
    print("Input images: ", img_paths)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_name = f'{sequence_name}_{frame_idx:06d}'
    vis_output_path = osp.join(output_dir, 'vis')
    Path(vis_output_path).mkdir(parents=True, exist_ok=True)

    # Parameters I am tuning
    human_loss_weight = 5.0
    stage2_start_idx_percentage = 0.2 #0.0 # 0.5 #0.2
    stage3_start_idx_percentage = 0.8 #0.9 
    min_niter = 500
    niter = 300
    niter_factor = 20 #20 ##10 # niter = int(niter_factor * scene_scale)
    lr = 0.01
    dist_tol = 0.3
    scale_increasing_factor = 2 #1.3
    num_of_humans_for_optimization = None
    focal_break = 20 # default is 20 in dust3r code
    shape_prior_weight = 1.0

    # Dust3r Config for the global alignment
    mode = GlobalAlignerMode.PointCloudOptimizer #if num_of_cams > 2 else GlobalAlignerMode.PairViewer
    device = 'cuda'
    silent = False
    schedule = 'linear'
    lr_base = lr
    lr_min = lr * 0.01 #0.0001
    init = 'known_params_hongsuk'
    niter_PnP = 10
    min_conf_thr_for_pnp = 3
    norm_pw_scale = False


    # 1) dust3r_network_output
    with open(dust3r_network_output_path, 'rb') as f:
        dust3r_network_output_dict = pickle.load(f)
    cam_names = dust3r_network_output_dict['img_names']
    # assert cam_names == [osp.basename(vitpose_path).split('__vitpose')[0] for vitpose_path in vitpose_paths] == [osp.basename(bytetrack_path).split('__bbox_with_score_used')[0] for bytetrack_path in bytetrack_paths]
    dust3r_network_output = dust3r_network_output_dict['output']
    affine_matrices = dust3r_network_output_dict['affine_matrices'] # list of N (2,3) affine matrices that transform from the original image size to the dust3r resized image size
    # Get inverse affine matrices to transform from dust3r resized image size back to original image size
    inv_affine_matrices = []
    for affine_matrix in affine_matrices:
        # Convert 2x3 affine matrix to 3x3 homogeneous form
        affine_matrix_3x3 = np.vstack([affine_matrix, [0, 0, 1]])
        # Get inverse
        inv_affine_matrix = np.linalg.inv(affine_matrix_3x3)
        # Convert back to 2x3 form
        inv_affine_matrix = inv_affine_matrix[:2]
        inv_affine_matrices.append(inv_affine_matrix)

    
    # 2) im_focals (K), im_poses (T)
    with open(vitpose_and_gt_path, 'rb') as f:
        vitpose_and_gt_dict = pickle.load(f)
    import pdb; pdb.set_trace()

    # show_ground_truth_scene(vitpose_and_gt_dict)

    # with open(prefit_path, 'rb') as f:
    #     prefit = pickle.load(f)
    # adjust the niter according to the scene scale
    scene_scale = vitpose_and_gt_dict['alpha'].cpu().numpy()  #prefit['cam']['alpha']
    niter = max(int(niter_factor * scene_scale), min_niter)
    scene_scale = scene_scale * scale_increasing_factor
    print(f"Adjusted niter to {niter} according to the scene scale {scene_scale} * {niter_factor}")

    im_intrinsics = vitpose_and_gt_dict['K'].cpu().numpy() # these are original image space intrinsics #prefit['cam']['K'] # (N, 3, 3)
    im_poses = vitpose_and_gt_dict['T'].cpu().numpy() # these are dust3r extrinsics #prefit['cam']['T'] # (N, 4, 4)

    # Apply scene scale to the extrinsics
    im_poses[:, :3, 3] = im_poses[:, :3, 3] * scene_scale

    # Apply inverse affine matrices to intrinsics
    # For each camera, multiply the intrinsic matrix by the inverse affine matrix
    # This inv affine matrix transforms the intrinsics from the original image space to the dust3r resized space
    transformed_intrinsics = []
    for i, (K, inv_affine) in enumerate(zip(im_intrinsics, inv_affine_matrices)):
        # Convert inv_affine to 3x3 by adding [0,0,1] row
        inv_affine_3x3 = np.vstack([inv_affine, [0, 0, 1]])
        # Transform intrinsics: K_orig = inv(A) * K
        # Where A is the affine transform from original to dust3r space
        K_transformed = inv_affine_3x3 @ K
        transformed_intrinsics.append(K_transformed)    
    # Update im_intrinsics with transformed values
    im_intrinsics = transformed_intrinsics

    # # TEMP; JUST FOR SAKE OF SAVING
    # im_poses_tmp = im_poses.copy()
    # im_poses_tmp[:, :3, 3] = im_poses_tmp[:, :3, 3] / scene_scale
    # im_intrinsics_tmp = im_intrinsics.copy()
    # # Initialize the scene optimizer
    # scene_tmp = global_aligner(dust3r_network_output, device=device, mode=mode, verbose=not silent, focal_break=focal_break, has_human_cue=False)
    # scene_tmp.norm_pw_scale = norm_pw_scale

    # # initialize the scene parameters with the known poses or point clouds
    # num_of_cams = len(cam_names)
    # if num_of_cams >= 2:
    #     if init == 'known_params_hongsuk':
    #         # im_focals: np.array to list of scalars
    #         im_focals_tmp = [im_intrinsics_tmp[i][0][0] for i in range(len(cam_names))]
    #         # im_poses: numpy to torch
    #         im_poses_tmp = [torch.from_numpy(im_poses_tmp[i]).to(device) for i in range(len(cam_names))]
    #         im_poses_tmp = torch.stack(im_poses_tmp)
            
    #         init_loss = scene_tmp.init_from_known_params_hongsuk(im_focals=im_focals_tmp, im_poses=im_poses_tmp, pts3d=None, niter_PnP=niter_PnP, min_conf_thr=min_conf_thr_for_pnp)

    # to_save_no_scale_dust3r_ga_output = parse_to_save_data(scene_tmp, cam_names)
    # # TEMP; JUST FOR SAKE OF SAVING

    # Initialize the scene optimizer
    scene = global_aligner(dust3r_network_output, device=device, mode=mode, verbose=not silent, focal_break=focal_break, has_human_cue=False)
    scene.norm_pw_scale = norm_pw_scale

    # initialize the scene parameters with the known poses or point clouds
    num_of_cams = len(cam_names)
    if num_of_cams >= 2:
        if init == 'known_params_hongsuk':
            # im_focals: np.array to list of scalars
            im_focals = [im_intrinsics[i][0][0] for i in range(len(cam_names))]
            # im_poses: numpy to torch
            im_poses = [torch.from_numpy(im_poses[i]).to(device) for i in range(len(cam_names))]
            im_poses = torch.stack(im_poses)
            
            init_loss = scene.init_from_known_params_hongsuk(im_focals=im_focals, im_poses=im_poses, pts3d=None, niter_PnP=niter_PnP, min_conf_thr=min_conf_thr_for_pnp)
            
            # TEMP
            if init_loss < 1:
                lr = lr * 0.1
                lr_base = lr
                lr_min = lr * 0.01
                print(f"Adjusted lr to {lr} because the init loss {init_loss:.3f} is low")

            # # TEMP; use gt focal lengthes and affine transform (divide by 7.5) and make it fixed
            # im_focals = [world_gt_cameras[cam_name]['K'][0] / 7.5 for cam_name in cam_names]
            # scene.preset_focal(im_focals, msk=None)
            # scene.im_focals.requires_grad = False

            print("Known params init")
        else:
            raise ValueError(f"Unknown initialization method: {init}")

    scene_params = [p for p in scene.parameters() if p.requires_grad]

    scaled_dust3r_ga_output = parse_to_save_data(scene, cam_names)
    # Visualize the intialized environment
    # show_env_in_viser(world_env=scaled_dust3r_ga_output, world_scale_factor=1.)

    # 3) human parameters
    # TEMP; hardcoded for aria01
    prefit = {
        'human': {
            'body_pose': vitpose_and_gt_dict['body_pose'].cpu().numpy(),
            'global_orient': vitpose_and_gt_dict['global_orient'].cpu().numpy(),
            'betas': vitpose_and_gt_dict['betas'].cpu().numpy(),
            'left_hand_pose': vitpose_and_gt_dict['left_hand_pose'].cpu().numpy(),
            'right_hand_pose': vitpose_and_gt_dict['right_hand_pose'].cpu().numpy(),
            'transl': vitpose_and_gt_dict['transl'].cpu().numpy(),
        }
    }
    human_params = {'aria01': prefit['human']}

    smplx_layer_dict = {
        # key: number of humans
        1: smplx.create(model_path = '/home/hongsuk/projects/egoexo/essentials/body_models', 
                        model_type = 'smplx', 
                        gender = 'neutral', 
                        # age = 'kid',
                        use_pca = False, 
                        num_pca_comps = 45, 
                        flat_hand_mean = True, 
                        use_face_contour = True, 
                        num_betas = 10, 
                        batch_size = 1).to(device)
    }
    new_human_params = {'aria01': {}}
    new_human_params['aria01']['body_pose'] = torch.from_numpy(human_params['aria01']['body_pose']).reshape(1, -1).to(device)
    new_human_params['aria01']['global_orient'] = torch.from_numpy(human_params['aria01']['global_orient']).reshape(1, -1).to(device)
    new_human_params['aria01']['betas'] = torch.from_numpy(human_params['aria01']['betas']).reshape(1, -1).to(device)
    new_human_params['aria01']['left_hand_pose'] = torch.from_numpy(human_params['aria01']['left_hand_pose']).reshape(1, -1).to(device)
    new_human_params['aria01']['right_hand_pose'] = torch.from_numpy(human_params['aria01']['right_hand_pose']).reshape(1, -1).to(device)
    new_human_params['aria01']['transl'] = torch.from_numpy(human_params['aria01']['transl']).reshape(1, -1).to(device)
    human_params = new_human_params

    if vis:
        show_optimization_results(scaled_dust3r_ga_output, human_params, smplx_layer_dict[1])


    # 4) load the multiview_multiperson_poses2d, multiview_multiperson_bboxes from ViTPose output, which are the targets for optimization
    # multiview_multiperson_poses2d: Dict[human_name -> Dict[cam_name -> (J, 3)]]
    # multiview_multiperson_bboxes: Dict[human_name -> Dict[cam_name -> (5)]]

    multiview_multiperson_poses2d = {'aria01': {}}
    multiview_multiperson_bboxes = {'aria01': {}}
    # for vitpose_path, bytetrack_path in zip(vitpose_paths, bytetrack_paths):
    #     with open(vitpose_path, 'rb') as f:
    #         # TEMP; hardcoded for aria01
    #         pose2d = pickle.load(f)[frame_idx][0] # (1, J, 3) -> (J, 3)
    #         multiview_multiperson_poses2d['aria01'][osp.basename(vitpose_path).split('__vitpose')[0]] = torch.from_numpy(pose2d).to(device)
    #     with open(bytetrack_path, 'rb') as f:
    #         # TEMP; hardcoded for aria01
    #         bbox = pickle.load(f)[frame_idx][0] # (1, 5) -> (5)
    #         multiview_multiperson_bboxes['aria01'][osp.basename(bytetrack_path).split('__bbox_with_score_used')[0]] = torch.from_numpy(bbox).to(device)
    
    # load vitpose from cleaned data; and also load the ground truth joints2d
    # with open(vitpose_and_gt_path, 'rb') as f:
    #     vitpose_and_gt_dict = pickle.load(f)
    # import pdb; pdb.set_trace()
    vitposeplus_2d_keypoints = vitpose_and_gt_dict['keypoints'] # (1, num_cams, 1, 135, 3)
    vitposeplus_2d_keypoints = vitposeplus_2d_keypoints[0, :, 0].numpy() # (num_cams, 135, 3)
    # change the joint order
    coco_wholebody_2d_keypoints = np.zeros((len(cam_names), len(COCO_WHOLEBODY_KEYPOINTS), 3)) # (num_cams, 133, 3)
    # For each camera and joint where mapper is not None, copy the keypoint data
    for coco_idx, vitpose_idx in enumerate(vitposeplus_to_coco_mapper):
        if vitpose_idx is not None:
            coco_wholebody_2d_keypoints[:, coco_idx, :] = vitposeplus_2d_keypoints[:, vitpose_idx, :]
    joints2d = coco_wholebody_2d_keypoints

    # Vis: draw_joints2d(joints2d, img_paths=img_paths, output_dir='./', conf_threshold=0.1)

    # Map the joints2d to the dust3r resized space
    for cam_idx, inv_affine in enumerate(inv_affine_matrices):
        homo_joints2d = np.concatenate([joints2d[cam_idx, :, :2], np.ones((joints2d.shape[1], 1))], axis=1)
        joints2d[cam_idx, :, :2] = (inv_affine @ homo_joints2d.T)[:2].T

    # get bounding boxes from the joints2d
    bboxes = get_bboxes_from_joints2d(joints2d) # (num_cams, 5)
    # TEMP
    # just assign 1
    bboxes[:, 4] = 1.
    for cam_name in cam_names:
        multiview_multiperson_poses2d['aria01'][cam_name] = torch.from_numpy(joints2d[cam_names.index(cam_name)]).to(device)
        multiview_multiperson_bboxes['aria01'][cam_name] = torch.from_numpy(bboxes[cam_names.index(cam_name)]).to(device)

    # 5) Final optimization
    # Logistics 
    save_2d_pose_vis = 10 
    scene_loss_timer = Timer()
    human_loss_timer = Timer()
    gradient_timer = Timer()

    # 1st stage; stage 1 is from 0% to 30%
    stage1_iter = list(range(0, int(niter * stage2_start_idx_percentage)))
    # 2nd stage; stage 2 is from 30% to 60%
    stage2_iter = list(range(int(niter * stage2_start_idx_percentage), int(niter * stage3_start_idx_percentage)))
    # 3rd stage; stage 3 is from 60% to 100%
    stage3_iter = list(range(int(niter * stage3_start_idx_percentage), niter))

    print(">>> Set the scene scale as a parameter to optimize")
    residual_scene_scale = nn.Parameter(torch.tensor(1., requires_grad=True).to(device))

    human_params = init_human_params(human_params, device)
    init_human_cam_data = {
        'human_params': copy.deepcopy(human_params),
        # 'human_inited_cam_poses': copy.deepcopy(human_inited_cam_poses),
    }
    if num_of_humans_for_optimization is None:
        num_of_humans_for_optimization = len(human_params)
        print(f"Optimizing all {num_of_humans_for_optimization} humans")
    else:
        num_of_humans_for_optimization = min(num_of_humans_for_optimization, len(human_params))
        print(f"Optimizing {num_of_humans_for_optimization} humans")
        print(f"Names of humans to optimize: {sorted(list(human_params.keys()))[:num_of_humans_for_optimization]}")


    # Given the number of iterations, run the optimizer while forwarding the scene with the current parameters to get the loss
    with tqdm.tqdm(total=niter) as bar:
        while bar.n < bar.total:
            # Set optimizer
            if len(stage1_iter) > 0 and bar.n == stage1_iter[0]:
                optimizer = get_stage_optimizer(human_params, scene_params, residual_scene_scale, 1, lr)
                print("\n1st stage optimization starts at ", bar.n)
            elif len(stage2_iter) > 0 and bar.n == stage2_iter[0]:
                # human_loss_weight = 10.
                human_loss_weight = 1.
                # TEMP
                # lr_base = lr = 0.01
                optimizer = get_stage_optimizer(human_params, scene_params, residual_scene_scale, 2, lr)
                print("\n2nd stage optimization starts at ", bar.n)
                # Reinitialize the scene
                print("Residual scene scale: ", residual_scene_scale.item())
                scene_intrinsics = scene.get_intrinsics().detach().cpu().numpy()
                im_focals = [intrinsic[0,0] for intrinsic in scene_intrinsics]
                im_poses = scene.get_im_poses().detach()
                im_poses[:, :3, 3] = im_poses[:, :3, 3] * residual_scene_scale.item()
                # pts3d = scene.get_pts3d()
                # pts3d_scaled = [p * residual_scene_scale.item() for p in pts3d]
                scene.init_from_known_params_hongsuk(im_focals=im_focals, im_poses=im_poses, pts3d=None, niter_PnP=niter_PnP, min_conf_thr=min_conf_thr_for_pnp)
                print("Known params init")
                    
                if False and vis:
                    # Visualize the initilization of 3D human and 3D world
                    world_env = parse_to_save_data(scene, cam_names)
                    show_optimization_results(world_env, human_params, smplx_layer_dict[1])

            elif len(stage3_iter) > 0 and bar.n == stage3_iter[0]:
                # human_loss_weight = 5.
                human_loss_weight = 1.
                
                optimizer = get_stage_optimizer(human_params, scene_params, residual_scene_scale, 3, lr)
                print("\n3rd stage optimization starts at ", bar.n)

                if False and vis:
                    # Visualize the initilization of 3D human and 3D world
                    world_env = parse_to_save_data(scene, cam_names)
                    show_optimization_results(world_env, human_params, smplx_layer_dict[1])

            lr = adjust_lr(bar.n, niter, lr_base, lr_min, optimizer, schedule)
            optimizer.zero_grad()

            # get extrinsincs and intrinsics from the scene
            multiview_cam2world_4by4  = scene.get_im_poses()  # (len(cam_names), 4, 4)

            if bar.n in stage1_iter:
                multiview_cam2world_4by4 = multiview_cam2world_4by4.detach()
                # Create a new tensor instead of modifying in place
                multiview_cam2world_3by4 = torch.cat([
                    multiview_cam2world_4by4[:, :3, :3],
                    (multiview_cam2world_4by4[:, :3, 3] * residual_scene_scale).unsqueeze(-1)
                ], dim=2)
                multiview_cam2world_4by4 = torch.cat([
                    multiview_cam2world_3by4,
                    multiview_cam2world_4by4[:, 3:4, :]
                ], dim=1)

            multiview_world2cam_4by4 = torch.inverse(multiview_cam2world_4by4) # (len(cam_names), 4, 4)
            multiview_intrinsics = scene.get_intrinsics().detach() # (len(cam_names), 3, 3)

            # Initialize losses dictionary
            losses = {}

            # Get human loss
            human_loss_timer.tic()
            losses['human_loss'], projected_joints = get_human_loss(smplx_layer_dict, num_of_humans_for_optimization, human_params, cam_names, 
                                                                    multiview_world2cam_4by4, multiview_intrinsics, 
                                                                    multiview_multiperson_poses2d, multiview_multiperson_bboxes, 
                                                                    shape_prior_weight, device)
            losses['human_loss'] = human_loss_weight * losses['human_loss']
            human_loss_timer.toc()

            if num_of_cams > 2 and (bar.n in stage2_iter or bar.n in stage3_iter):
                # Get scene loss
                scene_loss_timer.tic()
                losses['scene_loss'] = scene.dust3r_loss()
                scene_loss_timer.toc()  

            # Compute total loss
            total_loss = sum(losses.values())

            gradient_timer.tic()
            total_loss.backward()
            optimizer.step()
            gradient_timer.toc()

            # Create loss string for progress bar
            loss_str = f'{lr=:g} '
            loss_str += ' '.join([f'{k}={v:g}' for k, v in losses.items()])
            loss_str += f' total_loss={total_loss:g}'
            bar.set_postfix_str(loss_str)
            bar.update()

            if bar.n == 1 or bar.n == bar.total or vis and bar.n % save_2d_pose_vis == 0:
                for cam_idx, (cam_name, human_joints) in enumerate(projected_joints.items()):
                    img = scene.imgs[cam_names.index(cam_name)].copy() * 255.
                    img = img.astype(np.uint8)
                    for human_name, joints in human_joints.items():
                        # darw the human name
                        img = cv2.putText(img, human_name, (int(joints[0, 0]), int(joints[0, 1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        for idx, (joint, gt_joint) in enumerate(zip(joints, joints2d[cam_idx])):
                            img = cv2.circle(img, (int(joint[0]), int(joint[1])), 1, (0, 255, 0), -1)
                            # draw the gt joint
                            if gt_joint[2] > 0.1:
                                img = cv2.circle(img, (int(gt_joint[0]), int(gt_joint[1])), 1, (0, 0, 255), -1)

                    cv2.imwrite(osp.join(vis_output_path, f'{sequence_name}_{frame_idx:05d}_{cam_name}_{bar.n}.png'), img[:, :, ::-1])
    
    print("Final losses:", ' '.join([f'{k}={v.item():g}' for k, v in losses.items()]))
    print(f"Time taken: human_loss={human_loss_timer.total_time:g}s, scene_loss={scene_loss_timer.total_time:g}s, backward={gradient_timer.total_time:g}s")

    # Save output
    total_output = {}
    # From Lea
    total_output['gt_scene_xyz'] = vitpose_and_gt_dict['gt_scene_xyz'].cpu().numpy()
    total_output['gt_cam_T'] = vitpose_and_gt_dict['gt_cam_T'].cpu().numpy()
    total_output['gt_cam_K'] = vitpose_and_gt_dict['gt_cam_K'].cpu().numpy()
    # From Hongsuk
    total_output['our_pred_world_cameras_and_structure'] = parse_to_save_data(scene, cam_names)
    total_output['our_pred_humans_smplx_params'] = convert_human_params_to_numpy(human_params)
    total_output['dust3r_pred_world_cameras_and_structure'] = scaled_dust3r_ga_output #to_save_no_scale_dust3r_ga_output 
    total_output['hmr2_pred_humans_and_cameras'] = init_human_cam_data 
    total_output['our_optimized_human_names'] = sorted(list(human_params.keys()))[:num_of_humans_for_optimization]

    # # Create output name with final losses
    # output_name += '_final_loses'
    # for k, v in losses.items():
    #     output_name += f'_{k}_{v.item():.2f}'
    # print("Saving to ", osp.join(output_dir, f'{output_name}.pkl'))
    # with open(osp.join(output_dir, f'{output_name}.pkl'), 'wb') as f:
    #     pickle.dump(total_output, f)    

    if vis:
        show_optimization_results(parse_to_save_data(scene, cam_names), human_params, smplx_layer_dict[1])

if __name__ == '__main__':
    tyro.cli(main)



