import os
import os.path as osp
import pickle
import numpy as np
import tyro
import viser
import time
import cv2
import torch
import smplx

from scipy.spatial.transform import Rotation as R

from multihmr.blocks import SMPL_Layer


def main(world_env_pkl: str, world_scale_factor: float = 1., after_opt: bool = False):
    with open(world_env_pkl, 'rb') as f:
        world_env = pickle.load(f)

    smplx_layer = SMPL_Layer(type='smplx', gender='neutral', num_betas=10, kid=False, person_center='head')
    
    smplx_vertices_dict = {}
    if not after_opt:
        for img_name in world_env.keys():
            if img_name == 'non_img_specific':
                continue
        
            human_data = world_env[img_name]['multihmr_3d_outputs']
            # make the input to batch tensor with batch_size one
            # define new variables for the input
            transl = torch.from_numpy(human_data['transl']).unsqueeze(0) # (1,3)
            rotvec = torch.from_numpy(human_data['rotvec']).unsqueeze(0) # (1,53,3)
            shape = torch.from_numpy(human_data['shape']).unsqueeze(0) # (1,10)
            expression = torch.from_numpy(human_data['expression']).unsqueeze(0) # (1,10)
            K = torch.from_numpy(world_env[img_name]['intrinsic']).unsqueeze(0) # (1,3,3)

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
    else:
        smplx_params= world_env['non_img_specific']['smplx_params']

        # set the device of smplx_layer to the same device as smplx_params
        smplx_layer = smplx_layer.to(smplx_params['transl'].device)
        pose = torch.cat((smplx_params['global_rotvec'], smplx_params['relative_rotvec']), dim=1) # (1, 53, 3)
        smplx_output = smplx_layer(transl=smplx_params['transl'],
                                pose=pose,
                                shape=smplx_params['shape'],
                                K=torch.zeros((len(pose), 3, 3), device=pose.device),  # dummy
                                expression=smplx_params['expression'],
                                loc=None,
                                dist=None)
        smplx_vertices = smplx_output['v3d']
        smplx_vertices_dict = {
            'world': smplx_vertices.detach().squeeze().cpu().numpy()
        }
        smplx_faces = smplx_layer.bm_x.faces

    show_env_human_in_viser(world_env_pkl, world_scale_factor=world_scale_factor, smplx_vertices_dict=smplx_vertices_dict, smplx_faces=smplx_layer.bm_x.faces)

#     show_env_in_viser(world_env_pkl, world_scale_factor)

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
    
def show_env_human_in_viser(world_env_pkl: str, world_scale_factor: float = 1., smplx_vertices_dict: dict = None, smplx_faces: np.ndarray = None):
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
    for img_idx, img_name in enumerate(world_env.keys()):
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
            point_size=0.05,
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

    if smplx_vertices_dict is not None:
        for img_idx, img_name in enumerate(smplx_vertices_dict.keys()):
            vertices = smplx_vertices_dict[img_name]
            vertices = vertices @ rot_180       
            server.scene.add_mesh_simple(
                f"/{img_name}_main_human/mesh",
                vertices=vertices,
                faces=smplx_faces,
                flat_shading=False,
                wireframe=False,
                color=get_color(img_idx),
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

if __name__ == '__main__':
    tyro.cli(main)
