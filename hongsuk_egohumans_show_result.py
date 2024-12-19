import pickle
import tyro
import numpy as np
import smplx
import torch
from hongsuk_vis_viser_env_human import show_env_human_in_viser, visualize_cameras_and_human
from scipy.spatial.transform import Rotation
from scipy.linalg import orthogonal_procrustes


def show_optimization_results(world_env, human_params, smplx_layer, gt_cameras=None):
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
    show_env_human_in_viser(world_env=world_env, world_scale_factor=1., smplx_vertices_dict=smplx_vertices_dict, smplx_faces=smplx_layer.faces, gt_cameras=gt_cameras)


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


def procrustes_alignment(source_points, target_points):
    """
    Compute Procrustes alignment between two sets of points.
    Returns scale, rotation matrix, and translation vector.
    """
    # Center the points
    source_mean = np.mean(source_points, axis=0)
    target_mean = np.mean(target_points, axis=0)
    
    source_centered = source_points - source_mean
    target_centered = target_points - target_mean
    
    # Compute scale
    source_norm = np.linalg.norm(source_centered, ord='fro')
    target_norm = np.linalg.norm(target_centered, ord='fro')
    scale = target_norm / source_norm
    
    # Compute rotation
    R, _ = orthogonal_procrustes(source_centered, target_centered)
    
    # Compute translation
    t = target_mean - scale * (R @ source_mean)
    
    return scale, R, t


def main(optim_output_pkl: str):
    device = 'cuda'
    smplx_layer = smplx.create(model_path = '/home/hongsuk/projects/egoexo/essentials/body_models', model_type = 'smplx', gender = 'neutral', use_pca = False, num_pca_comps = 45, flat_hand_mean = True, use_face_contour = True, num_betas = 10, batch_size = 1).to(device)
     
    with open(optim_output_pkl, 'rb') as f:
        optim_output = pickle.load(f)

    # convert to torch tensor from numpy for human params
    human_params = optim_output['our_pred_humans_smplx_params']
    for human_name, optim_target_dict in human_params.items():
        for key, value in optim_target_dict.items():
            human_params[human_name][key] = torch.from_numpy(value).to(device)

    world_env = optim_output['our_pred_world_cameras_and_structure']
    # inspect the world_env
    for cam_name in world_env.keys():
        pts3d = torch.tensor(world_env[cam_name]['pts3d']).to(device)
        # check nan
        if torch.isnan(pts3d).any():
            import pdb; pdb.set_trace()
        
        cam2world = torch.tensor(world_env[cam_name]['cam2world']).to(device)
        # check if the cam2world is valid
        if torch.isnan(cam2world).any():
            import pdb; pdb.set_trace()

    human_inited_cam_poses = optim_output["hmr2_pred_humans_and_cameras"]["human_inited_cam_poses"]
    print("Camera names of human_inited_cam_poses: ", human_inited_cam_poses.keys())

    # check the dust3r cam
    dust3r_cam = optim_output["dust3r_pred_world_cameras_and_structure"]
    for cam_name in dust3r_cam.keys():
        cam2world = dust3r_cam[cam_name]['cam2world']
        print("cam_name: ", cam_name, cam2world)

    # visualize the groundtruth
    smpl_layer = smplx.create('./models', "smpl") # for GT
    smpl_layer = smpl_layer.to(device).float()
    world_multiple_human_3d_annot = optim_output['gt_world_humans_smpl_params']
    world_gt_cameras = optim_output['gt_world_cameras']
    world_gt_colmap_pointcloud_xyz = optim_output['gt_world_structure']
    world_gt_colmap_pointcloud_rgb = np.zeros_like(world_gt_colmap_pointcloud_xyz)

    # Visualize the groundtruth human parameters and cameras
    world_gt_cam_poses = {}
    for cam_name in sorted(world_gt_cameras.keys()):
        cam2world = world_gt_cameras[cam_name]['cam2world_4by4']
        world_gt_cam_poses[cam_name] = cam2world
    
    try:
        vis_decode_human_params_and_cameras(world_multiple_human_3d_annot, world_gt_cam_poses, smpl_layer, world_gt_colmap_pointcloud_xyz, world_gt_colmap_pointcloud_rgb, device)
    except:
        import pdb; pdb.set_trace()

    # show the results
    try:
        # world_pred_cam_poses = {}
        # for cam_name in sorted(world_gt_cameras.keys()):
        #     world_pred_cam_poses[cam_name] = world_env[cam_name]['cam2world']

        
        # # Extract camera positions
        # gt_cam_positions = np.array([pose[:3, 3] for pose in world_gt_cam_poses.values()])
        # pred_cam_positions = np.array([pose[:3, 3] for pose in world_pred_cam_poses.values()])
        
        # # Perform Procrustes alignment
        # scale, rotation, translation = procrustes_alignment(pred_cam_positions, gt_cam_positions)
        
        # print("Procrustes Alignment Results:")
        # print(f"Scale: {scale}")
        # print(f"Rotation:\n{rotation}")
        # print(f"Translation:\n{translation}")
        
        # # Optionally, apply the transformation to check alignment
        # aligned_pred_positions = scale * (pred_cam_positions @ rotation.T) + translation
        
        # # Compute alignment error
        # error = np.mean(np.linalg.norm(aligned_pred_positions - gt_cam_positions, axis=1))
        # print(f"Average alignment error: {error:.4f} units")

        # for cam_idx, cam_name in enumerate(sorted(world_gt_cameras.keys())):
        #     world_env[cam_name]['cam2world'][:3, 3] = aligned_pred_positions[cam_idx]

        show_optimization_results(world_env, human_params, smplx_layer, gt_cameras=world_gt_cam_poses)
    except:
        import pdb; pdb.set_trace()

if __name__ == '__main__':
    tyro.cli(main)


# /home/hongsuk/projects/dust3r/outputs/egohumans/2024nov14_good_cams_focal_fixed/num_of_cams2/002_legoassemble_400_cam03cam05.pkl
# /home/hongsuk/projects/dust3r/outputs/egohumans/2024nov14_good_cams_focal_fixed/num_of_cams2/003_legoassemble_100_cam03cam05.pkl
# /home/hongsuk/projects/dust3r/outputs/egohumans/2024nov14_good_cams_focal_fixed/num_of_cams2/005_legoassemble_200_cam03cam05.pkl
# /home/hongsuk/projects/dust3r/outputs/egohumans/2024nov14_good_cams_focal_fixed/num_of_cams2/006_legoassemble_0_cam03cam05.pkl