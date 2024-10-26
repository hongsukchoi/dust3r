import numpy as np
import torch
import torch.nn.functional as F
import cv2
import os
import pickle
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from multihmr.utils import get_smplx_joint_names
from multihmr.blocks import SMPL_Layer
from hongsuk_vis_viser_env_human import show_env_human_in_viser

# from multihmr outputs, 
# first just extract the main person manually, cam01: human index 0, cam02: human index 1, cam03: human index 4, cam04: human index 0
# for 2D outputs of multihmr, we have scores, 2D joints and bbox
# affine transformation is needed to align the 2D joints and bbox to the dust3r image
# for 3D outputs of multihmr, we have 'transl', 'transl_pelvis', 'rotvec', 'expression', 'shape', 'v3d', 'j3d',
# save rotvec, expression, and shape as it is 
# apply rigid transformation to transl to 'transl' according to the dust3r extrinsics
# throw away v3d and j3d, transl_pelvis. Just use them for getting the scale ratio and checking the alignment of (rotvec, expression, shape, transl)

def read_multihmr_outputs(multihmr_pkl):
    # Load multihmr data
    with open(multihmr_pkl, 'rb') as f:
        multihmr_data = pickle.load(f)

    multihmr_K = multihmr_data['K'][0] # (3,3)
    multihmr_humans_dict = multihmr_data['humans_dict']
    multihmr_affine_matrix_dict = multihmr_data['affine_matrix_dict']

    # filter out humans and just keep the main person
    main_human_dict = {}
    for img_idx, (img_name, humans) in enumerate(multihmr_humans_dict.items()):
        if 'cam01' in img_name:
            main_human = humans[0]
        elif 'cam02' in img_name:
            main_human = humans[1]
        elif 'cam03' in img_name:
            main_human = humans[4]
        elif 'cam04' in img_name:
            main_human = humans[0]
        
        main_human_dict[img_name] = main_human

    # new dictionary only with the main human, key is the img_name
    clean_multihmr_data = {}
    for img_name in main_human_dict.keys():
        clean_multihmr_data[img_name] = {
            'main_human': main_human_dict[img_name], # all human information
            'K': multihmr_K, # (3,3)
            'affine_matrix': multihmr_affine_matrix_dict[img_name] # (2,3)
        }

    return clean_multihmr_data

# from dust3r outputs, we have rgb images, world point maps, extrinsics, and intrinsics 
# we need to align the 3D points in the world point maps to the 3D points from multihmr
# Scale world point maps and extrinisics using the heuristic ratio between the 3D points in the world point maps and the 3D points from multihmr

def read_dust3r_outputs(dust3r_output_path):
    with open(dust3r_output_path, 'rb') as f:
        dust3r_data = pickle.load(f)

    return dust3r_data

def apply_affine_transformation(points, multihmr_affine_matrix, dust3r_affine_matrix):
    # make multihmr_affine_matrix (3,3)
    multihmr_affine_matrix = np.vstack([multihmr_affine_matrix, np.array([0, 0, 1])])
    # make dust3r_affine_matrix (3,3)
    dust3r_affine_matrix = np.vstack([dust3r_affine_matrix, np.array([0, 0, 1])])
    # make homogeneous points (N,3) from (N,2)
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))]).T

    # apply affine transformation
    points_transformed = np.linalg.inv(dust3r_affine_matrix) @ multihmr_affine_matrix @ points_homogeneous
    points_transformed = points_transformed[:2, :] / points_transformed[2, :]

    points_transformed = points_transformed.T # (J, 2)
    return points_transformed

def visualize_aligned_2d_outputs(score, bbox, j2d_transformed, dust3r_rgbimg):
    # Visualize the keypoints on the original image
    org_img = dust3r_rgbimg.copy() * 255.
    for joint in j2d_transformed:
        org_img = cv2.circle(org_img, (int(joint[0]), int(joint[1])), 3, (0, 255, 0), -1) 

    org_img = cv2.rectangle(org_img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255), 2)
    org_img = cv2.putText(org_img, f'{score:.2f}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return org_img


def align_multihmr_2d_outputs_to_dust3r_2d(multihmr_data, dust3r_data, output_dir, only_main_joints=True, vis=True):

    to_save_data = {}
    for img_name in multihmr_data.keys():
        multihmr_main_human = multihmr_data[img_name]['main_human']
        multihmr_affine_matrix = multihmr_data[img_name]['affine_matrix'] # (2,3)

        # get the rgb image from dust3r data    
        dust3r_rgbimg = dust3r_data[img_name]['rgbimg'] # (H,W,3)
        dust3r_affine_matrix = dust3r_data[img_name]['affine_matrix'] # (2,3)   

        # get the 2D human information from multihmr data
        human_det_score = multihmr_main_human['scores'] # (1,)
        human_bbox = multihmr_main_human['bbox'] # (4,)
        human_j2d = multihmr_main_human['j2d'] # (J, 2) 

        # apply affine transformation to the 2D joints and bbox so that it is aligned with the dust3r image
        human_j2d_transformed = apply_affine_transformation(human_j2d, multihmr_affine_matrix, dust3r_affine_matrix)
        # convert bbox from (x,y,w,h) to (x1,y1,x2,y2) of shape (2,2)
        human_bbox = np.array([[human_bbox[0], human_bbox[1]], [human_bbox[0] + human_bbox[2], human_bbox[1] + human_bbox[3]]])
        human_bbox_transformed = apply_affine_transformation(human_bbox, multihmr_affine_matrix, dust3r_affine_matrix)
        # convert back to (x,y,w,h) format
        human_bbox_transformed = np.array([human_bbox_transformed[0][0], human_bbox_transformed[0][1], human_bbox_transformed[1][0] - human_bbox_transformed[0][0], human_bbox_transformed[1][1] - human_bbox_transformed[0][1]])

        # visualize the aligned 2D outputs
        if vis:
            aligned_img = visualize_aligned_2d_outputs(human_det_score, human_bbox_transformed, human_j2d_transformed, dust3r_rgbimg)
            cv2.imwrite(os.path.join(output_dir, f'{img_name}.png'), aligned_img[..., ::-1])

        to_save_data[img_name] = {
            'human_det_score': human_det_score,
            'human_bbox': human_bbox_transformed,
            'human_j2d': human_j2d_transformed
        }
        if only_main_joints:
            # only keep the body joints excluding face and hand joints
            to_save_data[img_name]['human_j2d'] = human_j2d_transformed[:get_smplx_joint_names().index('jaw')]

    return to_save_data

def bilinear_interpolation(grid, pixels):
    """
    Perform bilinear interpolation to sample values from a grid given pixel coordinates
    using torch.nn.functional.grid_sample.
    
    Args:
    grid (torch.Tensor): Input grid of shape (H, W) containing floating point values.
    pixels (torch.Tensor): Pixel coordinates of shape (N, 2) containing floating point values.
    
    Returns:
    torch.Tensor: Sampled values of shape (N,) containing floating point values.
    """
    H, W = grid.shape
    N = pixels.shape[0]
    
    grid = grid.unsqueeze(0).unsqueeze(0) # Reshape grid to (1, 1, H, W) as expected by grid_sample
    pixels_normalized = 2 * pixels / torch.tensor([[W - 1, H - 1]]) - 1 # Normalize pixel coordinates to [-1, 1] range as expected by grid_sample
    pixels_normalized = pixels_normalized.unsqueeze(0).unsqueeze(2) # Reshape pixels to (1, N, 1, 2) as expected by grid_sample
    sampled_values = F.grid_sample(grid, pixels_normalized, mode='bilinear', align_corners=True) # Perform sampling
    
    return sampled_values.squeeze().numpy()

def get_multihmr_dust3r_scale_ratio(multihmr_2d_outputs_in_dust3r, multihmr_data, dust3r_data, only_main_joints=True, k=10):
    # get the scale ratio between the 3D points in the world point maps and the 3D points from multihmr

    sampled_depths_list = []
    sampled_confs_list = []
    sampled_multihmr_joint_depths = []
    for img_name in dust3r_data.keys():
        if img_name == 'non_img_specific':
            continue
        # get depth map and confidence map from dust3r data
        conf_map = dust3r_data[img_name]['conf'] # (H,W)
        pt3d = dust3r_data[img_name]['pts3d'] # (H*W, 3), 3d points in the world coordinate frame

        # get depth map from pt3d
        # transform the pointcloud to the camera coordinate frame
        cam2world = dust3r_data[img_name]['cam2world'] 
        world2cam = np.linalg.inv(cam2world)
        pts3d = np.dot(pt3d, world2cam[:3, :3].T) + world2cam[:3, 3]
        depth_map = pts3d[..., 2]
        img_size = dust3r_data[img_name]['rgbimg'].shape[:2]
        depth_map = depth_map.reshape(img_size)
        depth_map = depth_map.astype(np.float32) # (H,W)

        # sample depth values at the 2D joint locations using torch grid sample
        depth_map = torch.from_numpy(depth_map).float()
        conf_map = torch.from_numpy(conf_map).float()
        transformed_vertices = torch.from_numpy(multihmr_2d_outputs_in_dust3r[img_name]['human_j2d']).float()
        sampled_depths = bilinear_interpolation(depth_map, transformed_vertices)
        sampled_confs = bilinear_interpolation(conf_map, transformed_vertices)

        sampled_depths_list.append(sampled_depths)
        sampled_confs_list.append(sampled_confs)

        # get 3D joint depths from multihmr data
        human_j3d = multihmr_data[img_name]['main_human']['j3d'] # (J, 3)
        if only_main_joints:
            human_j3d = human_j3d[:get_smplx_joint_names().index('jaw')] # (J', 3)
        sampled_multihmr_joint_depths.append(human_j3d[:, 2])

    # flatten the lists to shape (N*J,)
    sampled_depths = np.concatenate(sampled_depths_list, axis=0).flatten() # (N*J,)
    sampled_confs = np.concatenate(sampled_confs_list, axis=0).flatten() # (N*J,)
    sampled_multihmr_joint_depths = np.concatenate(sampled_multihmr_joint_depths, axis=0).flatten() # (N*J,)

    # Filter out invalid depths (e.g., zero or negative values)
    valid_mask = (sampled_depths > 0) & (sampled_multihmr_joint_depths > 0)
    valid_depths = sampled_depths[valid_mask]
    valid_confs = sampled_confs[valid_mask]
    valid_multihmr_depths = sampled_multihmr_joint_depths[valid_mask]

    # Sort by confidence and get top-k indices
    # k = min(100, len(valid_confs))  # Use top 100 or all if less than 100
    top_k_indices = np.argsort(valid_confs)[-k:]

    # Get top-k depths and corresponding multihmr depths
    top_k_depths = valid_depths[top_k_indices]
    top_k_multihmr_depths = valid_multihmr_depths[top_k_indices]
    top_k_confs = valid_confs[top_k_indices]

    # Calculate ratios
    ratios = top_k_depths / top_k_multihmr_depths

    # Calculate weighted sum of ratios
    weights = top_k_confs / np.sum(top_k_confs)
    weighted_ratio = np.sum(ratios * weights)
    weighted_ratio = 1/weighted_ratio
    # calculate the weighted confidence
    weighted_conf = np.sum(top_k_confs * weights)

    print(f"Weighted scale ratio: {weighted_ratio:.4f}, Weighted confidence: {weighted_conf:.4f}")

    return weighted_ratio, weighted_conf

def scale_dust3r_outputs(dust3r_data, scale_ratio):
    # scale the pointcloud and extrinsics
    for img_name in dust3r_data.keys():
        if img_name == 'non_img_specific':
            continue
        pt3d = dust3r_data[img_name]['pts3d'] # (H*W, 3)
        pt3d = pt3d * scale_ratio
        dust3r_data[img_name]['pts3d'] = pt3d

        cam2world = dust3r_data[img_name]['cam2world']
        cam2world[:3, 3] = cam2world[:3, 3] * scale_ratio
        dust3r_data[img_name]['cam2world'] = cam2world

# transform transl, j3d, v3d from camera coordinate frame to world coordinate frame of dust3r
def transform_multihmr_outputs_to_dust3r_world(multihmr_data, dust3r_data):
    for img_name in multihmr_data.keys():
        human_data = multihmr_data[img_name]['main_human']
        cam2world = dust3r_data[img_name]['cam2world']
        human_data['transl'] = np.dot(human_data['transl'], cam2world[:3, :3].T) + cam2world[:3, 3]
        human_data['j3d'] = np.dot(human_data['j3d'], cam2world[:3, :3].T) + cam2world[:3, 3]
        human_data['v3d'] = np.dot(human_data['v3d'], cam2world[:3, :3].T) + cam2world[:3, 3]

        # apply cam2world rotation to the first rotvec, which is the global rotation of the body
        # human_data['rotvec'] is of shape (53, 3)
        # it is axis angle representation of rotation
        # convert it to rotation matrix and apply to the first rotation
        rotvec = human_data['rotvec'][0]
        rotmat = R.from_rotvec(rotvec).as_matrix()
        rotmat = np.dot(cam2world[:3, :3], rotmat)
        # convert to axis angle representation
        rotvec = R.from_matrix(rotmat).as_rotvec()
        human_data['rotvec'][0] = rotvec
    
# decode smplx mesh from the transformed multihmr data
def decode_smplx_mesh(multihmr_data, dust3r_data, output_dir=None, vis_2d=True):
    smplx_layer = SMPL_Layer(type='smplx', gender='neutral', num_betas=10, kid=False, person_center='head')
    
    smplx_vertices_dict = {}
    for img_name in multihmr_data.keys():
        human_data = multihmr_data[img_name]['main_human']
        # make the input to batch tensor with batch_size one
        # define new variables for the input
        transl = torch.from_numpy(human_data['transl']).unsqueeze(0) # (1,3)
        rotvec = torch.from_numpy(human_data['rotvec']).unsqueeze(0) # (1,53,3)
        shape = torch.from_numpy(human_data['shape']).unsqueeze(0) # (1,10)
        expression = torch.from_numpy(human_data['expression']).unsqueeze(0) # (1,10)
        K = torch.from_numpy(dust3r_data[img_name]['intrinsic']).unsqueeze(0) # (1,3,3)

        smplx_output = smplx_layer(transl=transl,
                                 pose=rotvec,
                                 shape=shape,
                                 K=K,
                                 expression=expression,
                                 loc=None,
                                 dist=None)

        vertices = smplx_output['v3d'].squeeze().detach().cpu().numpy()
        # faces = smpl_layer.bm_x.faces
        smplx_vertices_dict[img_name] = vertices

        if vis_2d:
            joints_2d = smplx_output['j2d'].squeeze().detach().cpu().numpy()
            rgbimg = dust3r_data[img_name]['rgbimg']
            tmp_img = rgbimg.copy() * 255.
            for joint in joints_2d:
                tmp_img = cv2.circle(tmp_img, (int(joint[0]), int(joint[1])), 3, (0, 255, 0), -1) 
            cv2.imwrite(os.path.join(output_dir, f'{img_name}_2d_joints_after_alignment.png'), tmp_img[..., ::-1])


    return smplx_vertices_dict, smplx_layer


def main():
    multihmr_pkl = '/home/hongsuk/projects/dust3r/outputs/egoexo/multihmr_data_egoexo.pkl'
    dust3r_output_path = '/home/hongsuk/projects/dust3r/outputs/egoexo/dust3r_reconstruction_results_egoexo.pkl'
    output_dir = '/home/hongsuk/projects/dust3r/outputs/egoexo/aligned_2d_outputs'

    Path(output_dir).mkdir(parents=True, exist_ok=True) 

    multihmr_data = read_multihmr_outputs(multihmr_pkl)
    dust3r_data = read_dust3r_outputs(dust3r_output_path)
    multihmr_2d_outputs_in_dust3r = align_multihmr_2d_outputs_to_dust3r_2d(multihmr_data, dust3r_data, output_dir, vis=True)
    scale_ratio, conf = get_multihmr_dust3r_scale_ratio(multihmr_2d_outputs_in_dust3r, multihmr_data, dust3r_data, only_main_joints=True, k=10)
    scale_dust3r_outputs(dust3r_data, scale_ratio) # in-place scaling
    transform_multihmr_outputs_to_dust3r_world(multihmr_data, dust3r_data) # in-place transformation

    # save the multihmr data with the scale ratio in the dust3r data
    for img_name in dust3r_data.keys():
        if img_name == 'non_img_specific':
            continue
        dust3r_data[img_name]['multihmr_2d_outputs'] = multihmr_2d_outputs_in_dust3r[img_name]
        dust3r_data[img_name]['multihmr_3d_outputs'] = multihmr_data[img_name]['main_human']
    dust3r_data['non_img_specific']['scale_ratio'] = scale_ratio

    new_dust3r_output_path = 'rescaled_' + os.path.basename(dust3r_output_path).replace('.pkl', '_with_multihmr_aligned.pkl')
    output_dir = os.path.dirname(dust3r_output_path)
    new_dust3r_output_path = os.path.join(output_dir, new_dust3r_output_path)
    with open(new_dust3r_output_path, 'wb') as f:
        pickle.dump(dust3r_data, f)
    print(f"Saved the rescaled dust3r outputs to {new_dust3r_output_path}")

    vis = True
    if vis:
        output_dir_smplx = '/home/hongsuk/projects/dust3r/outputs/egoexo/smplx_meshes'
        Path(output_dir_smplx).mkdir(parents=True, exist_ok=True) 
        smplx_vertices_dict, smplx_layer = decode_smplx_mesh(multihmr_data, dust3r_data, output_dir_smplx, vis_2d=True)
        show_env_human_in_viser(dust3r_output_path, world_scale_factor=scale_ratio, smplx_vertices_dict=smplx_vertices_dict, smplx_faces=smplx_layer.bm_x.faces)

if __name__ == '__main__':
    main()

