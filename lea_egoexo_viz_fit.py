import os
import pickle
import viser
import viser.transforms as vtf
import numpy as np 
import smplx 
import tyro


def rot_transl_align_points(P, Q):
    """
    Finds the transformation matrix to align two sets of 3D points.
    
    Parameters:
    P (numpy.ndarray): Nx3 array of source points
    Q (numpy.ndarray): Nx3 array of target points
    
    Returns:
    R (numpy.ndarray): 3x3 rotation matrix
    t (numpy.ndarray): 3x1 translation vector
    T (numpy.ndarray): 4x4 transformation matrix that combines rotation and translation
    """
    # Compute centroids
    centroid_P = np.mean(P, axis=0)
    centroid_Q = np.mean(Q, axis=0)
    
    # Center the points
    P_centered = P - centroid_P
    Q_centered = Q - centroid_Q
    
    # Compute the covariance matrix
    H = P_centered.T @ Q_centered
    
    # Perform Singular Value Decomposition (SVD)
    U, _, Vt = np.linalg.svd(H)
    
    # Compute the rotation matrix
    R = Vt.T @ U.T
    
    # Handle special case for reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute the translation vector
    t = centroid_Q - R @ centroid_P
    
    # Form the 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return R, t, T

def load_fitted_results(result_folder_base, take_name, frame):
    """
    Load fitted results from a specified result folder for a given take and frame.
    """
    result_folder_run = f'{result_folder_base}/{take_name}'
    fitted_result_path = f'{result_folder_run}/{frame}/results/prefit.pkl'
    fitted_result = pickle.load(open(fitted_result_path, 'rb'))
    return fitted_result

def load_ground_truth_results(result_folder_base, take_name, frame):
    """
    Load fitted results from a specified result folder for a given take and frame.
    """
    result_folder_run = f'{result_folder_base}/{take_name}'
    input_path = f'{result_folder_run}/{frame}/input_data.pkl'
    input_data = pickle.load(open(input_path, 'rb'))
    return input_data

def viz_fitted_results(
        fitted_result, 
        port='8097', 
        colors_path='llib/visualization/colors.txt', 
        mesh_as_point_cloud=False,
        body_model_path='./models',
        body_model_type='smplx',
        point_shape='circle',
        body_model_point_size=0.006,
        scene_point_size=0.006,
        ground_truth=None
    ):
    """
    Visualizes fitted SMPL-X model results using a specified viewer, with options for customization.
    
    Parameters:
    ----------
    fitted_result : dict
        The fitted result data containing SMPL-X model parameters and potentially scene-related information.
        E.g. from prefit.pkl.
    
    port : str, optional, default='8097'
        The port number for the visualization server, typically used for a local visualization web server.
    
    colors_path : str, optional, default='llib/visualization/colors.txt'
        The file path to a text file containing RGB color values for visualization.
    
    mesh_as_point_cloud : bool, optional, default=False
        If True, the SMPL-X model mesh will be visualized as a point cloud instead of a full mesh.
    
    smplx_model_path : str, optional, default='essentials/body_models'
        The directory path to the SMPL-X model files, used to load the 3D model structure.
    
    model_type : str, optional, default='smplx'
        The type of body model to visualize, default is 'smplx'. Other potential values could be 'smpl', 'smplh', etc.
    
    point_shape : str, optional, default='circle'
        The shape of points for visualization, options may include 'circle', 'square', etc.
    
    smpl_point_size : float, optional, default=0.006
        The size of points for the SMPL-X model if displayed as a point cloud.
    
    scene_point_size : float, optional, default=0.006
        The size of points for scene data if displayed in the visualization.

    ground_truth : dict, optional, default=None
        The ground truth data containing SMPL-X model parameters and potentially scene-related information.
    
    Returns:
    -------
    None
        This function does not return any values. It launches a visualization of the fitted SMPL-X model in viser.
    """

    # create viser server
    server = viser.ViserServer(port=port)
    server.scene.set_up_direction('-y')

    # create smplx model (to get the faces)
    if not mesh_as_point_cloud:
        bm = smplx.create(
            model_path=body_model_path, 
            model_type=body_model_type
        )

    # color palette for human meshes
    with open(colors_path, 'r') as f:
        colors = np.loadtxt(f)

    # add camera estimate 
    fitted_cams_extrins = np.array(fitted_result['cam']['T'])
    fitted_cams_intrins = np.array(fitted_result['cam']['K'])
    for cam_idx, cam_pose in enumerate(fitted_cams_extrins):
        # add camera axes
        server.scene.add_batched_axes(
            f'fitted-camera-{cam_idx}', 
            batched_wxyzs=np.array(
                # Convert Nx3x3 rotation matrices to Nx4 quaternions.
                [vtf.SO3.from_matrix(cam_pose[:3, :3]).wxyz]
            ),
            batched_positions=np.array([cam_pose[:3, 3]]),
            axes_length=1.0,
            axes_radius=0.01,
        )
        # add camera frustum
        fx = fitted_cams_intrins[cam_idx][0, 0]
        iw = fitted_cams_intrins[cam_idx][0, 2]
        ih = fitted_cams_intrins[cam_idx][1, 2]
        fov = 2 * np.arctan(0.5 * iw / fx) * 180 / np.pi
        server.scene.add_camera_frustum(
            f'fitted-camera-frustum-{cam_idx}',
            fov=fov,
            aspect= iw / ih,
            wxyz=vtf.SO3.from_matrix(cam_pose[:3, :3]).wxyz,
            position=cam_pose[:3, 3],
            color=[0, 1, 0]
        )

    # add dust3r points
    dust3r_points = fitted_result['scene']
    dust3r_points_xyz = dust3r_points['dust3r_scene_xyz'] #.cpu().numpy()
    server.scene.add_point_cloud(
        f'dust3r-scene-scaled',
        points=dust3r_points_xyz,
        colors=dust3r_points['dust3r_scene_rgb'],
        point_size=scene_point_size,
        point_shape=point_shape
    )

    dust3r_points_xyz_orig = dust3r_points_xyz / fitted_result['cam']['alpha']
    server.scene.add_point_cloud(
        f'dust3r-scene-original',
        points=dust3r_points_xyz_orig,
        colors=dust3r_points['dust3r_scene_rgb'],
        point_size=0.0006,
        point_shape=point_shape
    )

    # add the human meshes 
    fitted_meshes = fitted_result['human']['vertices']
    zero_position = np.eye(4)
    for mesh_idx, mesh in enumerate(fitted_meshes):
        if mesh_as_point_cloud:
            mesh = mesh[0]
            color_mesh = [np.ones_like(mesh) * colors[mesh_idx % len(colors)]][0] / 255
            server.scene.add_point_cloud(
                f'fitted-human-{mesh_idx}',
                points=mesh,
                point_size=body_model_point_size,
                point_shape=point_shape,
                colors=color_mesh
            )
        else:
            color_mesh = colors[mesh_idx % len(colors)] / 255 
            server.scene.add_mesh_simple(
                f'fitted-human-{mesh_idx}',
                vertices=mesh[0],
                faces=bm.faces.astype(int),
                position=zero_position[:3, 3],
                wxyz=vtf.SO3.from_matrix(zero_position[:3, :3]).wxyz,
                color=color_mesh
            )

    # add the ground truth scene, joints, and camera
    if ground_truth is not None:
        # rotation & translation-align ground truth and our estimate (by camera)
        RR, tt, TT = rot_transl_align_points(
            ground_truth['gt_cam_T'].detach().cpu().numpy().copy()[:,:3,3], 
            fitted_cams_extrins.copy()[:,:3,3], 
        )

        # add ground truth dust3r points
        gt_aria_points_xyz = ground_truth['gt_scene_xyz']
        gt_aria_points_xyz = np.einsum('ij,aj->ai', RR, gt_aria_points_xyz) + tt
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
        gt_extrins = TT @ gt_extrins
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

        # add the ground truth joints 
        masked_joints = ground_truth['gt_joints3d'][:25,:3][ground_truth['gt_joints3d'][:25,3] > 0]
        masked_joints = np.einsum('ij,aj->ai', RR, masked_joints) + tt
        joint_color = np.ones_like(masked_joints)
        joint_color[:,1] = 1.0
        server.scene.add_point_cloud(
            'gt-human-joints', 
            points=masked_joints, 
            point_size=0.1,
            point_shape="circle",
            colors=joint_color
        )

    # set break point for debugging
    import pdb; pdb.set_trace()
    print("Done visualizing")
    
def main(
    result_folder_base: str = '/scratch/partial_datasets/egoexo/egoexo4d_v2_mvopti/run_02/train', # args.result_folder_base
    take_name: str = 'iiith_cooking_80_2', #args.take_name
    frame: str = '1086', #args.frame
    port: str = '4859', #args.port
    prefit_path: str = '/scratch/partial_datasets/egoexo/egoexo4d_v2_mvopti/run_02/train/iiith_cooking_80_2/1086/results/prefit.pkl', #args.prefit
    ground_truth_path: str = '/scratch/partial_datasets/egoexo/egoexo4d_v2_mvopti/run_02/train/iiith_cooking_80_2/1086/input_data.pkl', #args.ground_truth
    mesh_as_point_cloud: bool = True, #args.mesh_as_point_cloud
    body_model_path: str = './models', #args.body_model_path
    body_model_type: str = 'smplx', # = args.body_model_type
    colors_path: str = './colors.txt', # = args.colors_path
    point_shape: str = 'circle',
    body_model_point_size: float = 0.006,
    scene_point_size: float = 0.006
):
    
    if prefit_path is not None:
        fitted_result = pickle.load(open(prefit_path, 'rb'))
    else:
        fitted_result = load_fitted_results(
            result_folder_base, take_name, frame
        )

    if ground_truth_path is not None:
        ground_truth = pickle.load(open(ground_truth_path, 'rb'))
    else:
        ground_truth = load_ground_truth_results(
            result_folder_base, take_name, frame
        )

    viz_fitted_results(
        fitted_result,         
        port=port, 
        colors_path=colors_path,
        mesh_as_point_cloud=mesh_as_point_cloud,
        body_model_path=body_model_path,
        body_model_type=body_model_type,
        point_shape=point_shape,
        body_model_point_size=body_model_point_size,
        scene_point_size=scene_point_size,
        ground_truth=ground_truth
    )

if __name__ == '__main__':
    tyro.cli(main)
