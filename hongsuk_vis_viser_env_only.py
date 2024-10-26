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


def main(world_env_pkl: str, world_scale_factor: float = 5.):
    show_env_in_viser(world_env_pkl, world_scale_factor)

def show_env_in_viser(world_env_pkl: str, world_scale_factor: float = 5.):
    # Extract data from world_env dictionary
    # Load world environment data estimated by Mast3r
    with open(world_env_pkl, 'rb') as f:
        world_env = pickle.load(f)


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
