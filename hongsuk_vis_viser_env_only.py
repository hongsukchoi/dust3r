import os
import os.path as osp
import pickle
import numpy as np
import tyro
import viser
import time
import cv2
import smplx

from scipy.spatial.transform import Rotation as R


def visualize_cameras(cam_poses):
    # set viser
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+y")

    # get rotation matrix of 180 degrees around x axis
    rot_180 = np.eye(3)
    rot_180[1, 1] = -1
    rot_180[2, 2] = -1  

    # Add GUI elements.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)
    cam_handles = []
    for cam_name, cam_pose in cam_poses.items():
        # Visualize the camera
        camera = cam_pose
        camera[:3, :3] = rot_180 @ camera[:3, :3] 
        camera[:3, 3] = camera[:3, 3] @ rot_180
        
        # rotation matrix to quaternion
        quat = R.from_matrix(camera[:3, :3]).as_quat()
        # xyzw to wxyz
        quat = np.concatenate([quat[3:], quat[:3]])
        # translation vector
        trans = camera[:3, 3]

        # add camera
        cam_handle = server.scene.add_frame(
            f"/cam_{cam_name}",
            wxyz=quat,
            position=trans,
            show_axes=True,
            axes_length=0.5,
            axes_radius=0.04,
        )
        cam_handles.append(cam_handle)

    # add transform controls, initialize the location with the first two cameras
    control0 = server.scene.add_transform_controls(
        "/controls/0",
        position=cam_handles[0].position,
        scale=cam_handles[0].axes_length,
    )
    control1 = server.scene.add_transform_controls(
        "/controls/1",
        position=cam_handles[1].position,
        scale=cam_handles[1].axes_length,
    )
    distance_text = server.gui.add_text("Distance", initial_value="Distance: 0")

    def update_distance():
        distance = np.linalg.norm(control0.position - control1.position)
        distance_text.value = f"Distance: {distance:.2f}"

        server.scene.add_spline_catmull_rom(
            "/controls/line",
            np.stack([control0.position, control1.position], axis=0),
            color=(255, 0, 0),
        )

    control0.on_update(lambda _: update_distance())
    control1.on_update(lambda _: update_distance())
    
    start_time = time.time()
    while True:
        time.sleep(0.01)
        timing_handle.value = (time.time() - start_time) 


def procrustes_align(X, Y):
    """
    Performs Procrustes alignment between two sets of points X and Y
    Returns scale, rotation, translation
    
    Args:
        X: Ground truth points (N x 3)
        Y: Points to be aligned (N x 3)
    Returns:
        scale: Scale factor
        R: Rotation matrix (3 x 3) 
        t: Translation vector (3,)
    """
    # Center the points
    muX = X.mean(axis=0)
    muY = Y.mean(axis=0)
    
    X0 = X - muX
    Y0 = Y - muY

    # Compute scale
    ssX = (X0**2).sum()
    ssY = (Y0**2).sum()
    scale = np.sqrt(ssX/ssY)
    
    # Scale points
    Y0 = Y0 * scale
    
    # Compute rotation
    U, _, Vt = np.linalg.svd(X0.T @ Y0)
    R = U @ Vt
    
    # Ensure right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[-1,:] *= -1
        R = U @ Vt
    
    # Compute translation
    t = muX - scale * (R @ muY)
    
    return scale, R, t

def main(world_env_pkl: str, world_scale_factor: float = 5.):
    show_env_in_viser(world_env_pkl, world_scale_factor)

def show_env_in_viser(world_env_pkl: str = '', world_env: dict = None, world_scale_factor: float = 5., gt_cameras: dict = None):
    if world_env is None:
        # Extract data from world_env dictionary
        print(f"Loading world environment data from {world_env_pkl}")
        with open(world_env_pkl, 'rb') as f:
            world_env = pickle.load(f)

    # If we have ground truth cameras, compute scale using Procrustes alignment
    if gt_cameras is not None:
        # Collect camera positions
        gt_positions = []
        est_positions = []
        
        for img_name in gt_cameras.keys():
            if img_name in world_env:
                # Get ground truth camera position
                gt_pos = gt_cameras[img_name]['cam2world_4by4'][:3, 3]
                gt_positions.append(gt_pos)
                
                # Get estimated camera position
                est_pos = world_env[img_name]['cam2world'][:3, 3]
                est_positions.append(est_pos)
        
        # Convert to numpy arrays
        gt_positions = np.array(gt_positions)
        est_positions = np.array(est_positions)
        
        # Perform Procrustes alignment
        scale, _, _ = procrustes_align(gt_positions, est_positions)
        print(f"Computed scale factor from Procrustes alignment: {scale:.4f}")
        
        # Use computed scale instead of provided world_scale_factor
        world_scale_factor = scale

    # Apply scaling
    for img_name in world_env.keys():
        if img_name == 'non_img_specific':
            continue
        world_env[img_name]['pts3d'] *= world_scale_factor
        world_env[img_name]['cam2world'][:3, 3] *= world_scale_factor
        # get new mask
        conf = world_env[img_name]['conf']
        world_env[img_name]['msk'] = conf > 1.5
    
    # set viser
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+y")

    # get rotation matrix of 180 degrees around x axis
    rot_180 = np.eye(3)
    rot_180[1, 1] = -1
    rot_180[2, 2] = -1

    # Add GUI elements.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)

    pointcloud_handles = []
    cam_handles = []
    for _, img_name in enumerate(world_env.keys()):
        if img_name == 'non_img_specific':
            continue
        # Visualize the pointcloud of environment
        pts3d = world_env[img_name]['pts3d']
        if pts3d.ndim == 3:
            pts3d = pts3d.reshape(-1, 3)
        points = pts3d[world_env[img_name]['msk'].flatten()]
        colors = world_env[img_name]['rgbimg'][world_env[img_name]['msk']].reshape(-1, 3)
        # # no masking
        # points = pts3d
        # colors =world_env[img_name]['rgbimg'].reshape(-1, 3)

        points = points @ rot_180
        pc_handle = server.scene.add_point_cloud(
            f"/pts3d_{img_name}",
            points=points,
            colors=colors,
            point_size=0.01,
            point_shape='circle'
        )
        pointcloud_handles.append(pc_handle)

        # Visualize the camera
        camera = world_env[img_name]['cam2world']
        camera[:3, :3] = rot_180 @ camera[:3, :3] 
        camera[:3, 3] = camera[:3, 3] @ rot_180
        
        # rotation matrix to quaternion
        quat = R.from_matrix(camera[:3, :3]).as_quat()
        # xyzw to wxyz
        quat = np.concatenate([quat[3:], quat[:3]])
        # translation vector
        trans = camera[:3, 3]

        # add camera
        cam_handle = server.scene.add_frame(
            f"/cam_{img_name}",
            wxyz=quat,
            position=trans,
            show_axes=True,
            axes_length=0.5,
            axes_radius=0.04,
        )
        cam_handles.append(cam_handle)

    if gt_cameras is not None:
        #  'cam01': {
        #         'cam2world_R': np.ndarray,  # shape (3, 3), rotation matrix
        #         'cam2world_t': np.ndarray,  # shape (3,), translation vector
        #         'K': np.ndarray,  # shape (3, 3), intrinsic matrix
        #         'img_width': int,
        #         'img_height': int
        #     },
        #     'cam02': {...},
        #     'cam03': {...},
        #     'cam04': {...}
        
        for img_name in gt_cameras.keys():
            # Visualize the gt camera
            camera = gt_cameras[img_name]
            cam2world_Rt_homo = camera['cam2world_4by4'].copy()

            cam2world_R = rot_180 @ cam2world_Rt_homo[:3, :3]
            cam2world_t = cam2world_Rt_homo[:3, 3] @ rot_180

            # rotation matrix to quaternion
            quat = R.from_matrix(cam2world_R).as_quat()
            # xyzw to wxyz
            quat = np.concatenate([quat[3:], quat[:3]])
            # translation vector
            trans = cam2world_t   

            # add camera
            cam_handle = server.scene.add_frame(
                f"/gt_cam_{img_name}",
                wxyz=quat,
                position=trans,
            )
            cam_handles.append(cam_handle)


    # add transform controls, initialize the location with the first two cameras
    control0 = server.scene.add_transform_controls(
        "/controls/0",
        position=cam_handles[0].position,
        scale=cam_handles[0].axes_length,
    )
    control1 = server.scene.add_transform_controls(
        "/controls/1",
        position=cam_handles[1].position,
        scale=cam_handles[1].axes_length,
    )
    distance_text = server.gui.add_text("Distance", initial_value="Distance: 0")


    def update_distance():
        distance = np.linalg.norm(control0.position - control1.position)
        distance_text.value = f"Distance: {distance:.2f}"

        server.scene.add_spline_catmull_rom(
            "/controls/line",
            np.stack([control0.position, control1.position], axis=0),
            color=(255, 0, 0),
        )

    control0.on_update(lambda _: update_distance())
    control1.on_update(lambda _: update_distance())

    start_time = time.time()
    while True:
        time.sleep(0.01)
        timing_handle.value = (time.time() - start_time) 


if __name__ == '__main__':
    tyro.cli(main)
