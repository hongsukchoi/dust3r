import tyro
import pickle
import os
import os.path as osp
import torch
import numpy as np
import smplx
import cv2
import viser
import time

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R   

def get_color(idx):
    colors = np.array([
        [255, 0, 0],
        [0, 255, 0],
        [0, 0, 255],
        [255, 255, 0],
        [255, 0, 255],
        [0, 255, 255],
    ])  
    return colors[idx % len(colors)]


def main():
    lea_fits_path = '/scratch/partial_datasets/egoexo/egoexo4d_v2_mvopti/debug_01/train/upenn_0629_Dance_3_5/debug_001/1002/results/prefit.pkl'

    with open(lea_fits_path, 'rb') as f:
        lea_fits = pickle.load(f)

    dust3r_output_path = '/scratch/partial_datasets/egoexo4d_v2/processed_20240718/takes/upenn_0629_Dance_3_5/preprocessing/dust3r_world_env/dust3r_world_env.pkl'
    dust3r_input_img_dir = '/scratch/partial_datasets/egoexo4d_v2/processed_20240718/takes/upenn_0629_Dance_3_5/preprocessing/dust3r_world_env/images_frame100' # cam01.png, cam02.png, ...
    with open(dust3r_output_path, 'rb') as f:
        dust3r_output = pickle.load(f)
    
    # Get camera parameters
    scale_from_dust3r_to_lea_fit = lea_fits['cam']['alpha'] # scalar
    optimized_cam_intrinsic = lea_fits['cam']['K'] # (num_cams, 3,3)
    optimized_cam_extrinsic = lea_fits['cam']['T'] # (num_cams, 4,4)

    # Get dust3r pointcloud
    dust3r_pointcloud = []
    dust3r_colors = []
    dust3r_cameras = []
    for cam_idx, img_name in enumerate(dust3r_output.keys()):
        # Visualize the pointcloud of environment
        pts3d = dust3r_output[img_name]['pts3d']
        if pts3d.ndim == 3:
            pts3d = pts3d.reshape(-1, 3)
        points = pts3d[dust3r_output[img_name]['msk'].flatten()]
        colors = dust3r_output[img_name]['rgbimg'][dust3r_output[img_name]['msk']].reshape(-1, 3)

        # scale the dust3r pointcloud and cameras
        points *= scale_from_dust3r_to_lea_fit
        dust3r_output[img_name]['cam2world'][:3, 3] *= scale_from_dust3r_to_lea_fit

        dust3r_pointcloud.append(points)
        dust3r_colors.append(colors)
        dust3r_cameras.append(dust3r_output[img_name]['cam2world'])

    # define the smplx layer
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

    # Upload body parameters
    body_pose = torch.from_numpy(lea_fits['human']['body_pose'].reshape(1, -1)).to('cuda')
    global_orient = torch.from_numpy(lea_fits['human']['global_orient'].reshape(1, -1)).to('cuda')
    betas = torch.from_numpy(lea_fits['human']['betas'].reshape(1, -1)).to('cuda')
    left_hand_pose = torch.from_numpy(lea_fits['human']['left_hand_pose'].reshape(1, -1)).to('cuda')
    right_hand_pose = torch.from_numpy(lea_fits['human']['right_hand_pose'].reshape(1, -1)).to('cuda')
    transl = torch.from_numpy(lea_fits['human']['transl'].reshape(1, -1)).to('cuda')

    # Compute the SMPLx body
    # smpl-x forward path
    smplx_layer = smplx_layer.to('cuda')

    body = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=transl)

    vertices_list = body.vertices.detach().cpu().numpy() # (1, 10475, 3)

    # Visualize in Viser
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+y")

    # get rotation matrix of 180 degrees around x axis
    rot_180 = np.eye(3)
    rot_180[1, 1] = -1
    rot_180[2, 2] = -1  

    # Add GUI elements.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)

    # Add the SMPLx body mesh
    for person_idx in range(vertices_list.shape[0]):
        vertices = vertices_list[person_idx] @ rot_180       
        server.scene.add_mesh_simple(
            f"/smplx_person{person_idx}/mesh",
            vertices=vertices,
            faces=smplx_layer.faces,
            flat_shading=False,
            wireframe=False,
            color=get_color(person_idx),
        )

    # Add optimized camera poses
    cam_handles = []
    for cam_idx in range(optimized_cam_extrinsic.shape[0]):
        # Visualize the gt camera
        cam2world_Rt_homo = optimized_cam_extrinsic[cam_idx]

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
            f"/cam_{cam_idx}",
            wxyz=quat,
            position=trans,
        )
        cam_handles.append(cam_handle)

    # Add dust3r pointcloud
    for cam_idx in range(len(dust3r_pointcloud)):
        server.scene.add_point_cloud(
            f"/dust3r_pointcloud_{cam_idx}",
            points=dust3r_pointcloud[cam_idx],
            colors=dust3r_colors[cam_idx],
        )


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

    print("done")


if __name__ == '__main__':
    tyro.cli(main)