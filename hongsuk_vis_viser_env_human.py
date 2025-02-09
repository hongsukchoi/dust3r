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
import copy

from scipy.spatial.transform import Rotation as R

from multihmr.blocks import SMPL_Layer



# Read colors from colors.txt file
colors_path = osp.join(osp.dirname(__file__), 'colors.txt')
colors = []
with open(colors_path, 'r') as f:
    for line in f:
        # Convert each line of RGB values to a list of integers
        rgb = list(map(int, line.strip().split()))
        colors.append(rgb)
COLORS = np.array(colors)

def get_color(idx):
    return COLORS[(idx + 10) % len(COLORS)]

def visualize_cameras_and_human(cam_poses, human_vertices, smplx_faces, world_colmap_pointcloud_xyz=None, world_colmap_pointcloud_rgb=None):
    cam_poses = copy.deepcopy(cam_poses)
    human_vertices = copy.deepcopy(human_vertices)
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

    human_idx = 0
    for human_name in sorted(human_vertices.keys()):
        vertices = human_vertices[human_name] @ rot_180       
        server.scene.add_mesh_simple(
            f"/{human_name}_human/mesh",
            vertices=vertices,
            faces=smplx_faces,
            flat_shading=False,
            wireframe=False,
            color=get_color(human_idx),
        )
        human_idx += 1


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

    # Add scene structure pointcloud
    if world_colmap_pointcloud_xyz is not None:
        world_colmap_pointcloud_xyz = world_colmap_pointcloud_xyz @ rot_180
        server.scene.add_point_cloud(
        "/world_colmap_pointcloud",
        points=world_colmap_pointcloud_xyz,
        colors=world_colmap_pointcloud_rgb,
        point_size=0.01,
        point_shape='circle'
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
    
def show_env_human_in_viser(world_env: dict = None, world_env_pkl: str = '', world_scale_factor: float = 1., smplx_vertices_dict: dict = None, smplx_faces: np.ndarray = None, gt_cameras: dict = None):
    if world_env is None:
        # Load world environment data estimated by Mast3r
        with open(world_env_pkl, 'rb') as f:
            world_env = pickle.load(f)
    
    for img_name in world_env.keys():
        if img_name == 'non_img_specific':
            continue
        world_env[img_name]['pts3d'] *= world_scale_factor
        world_env[img_name]['cam2world'][:3, 3] *= world_scale_factor
        # get new mask
        # conf = world_env[img_name]['conf']
        # world_env[img_name]['msk'] = conf > 1.5
    
    # set viser
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    server.scene.set_up_direction("+y")

    needs_update: bool = True

    def set_stale(_) -> None:
        nonlocal needs_update
        needs_update = True

    # frustum gui elements
    gui_line_width = server.gui.add_slider(
        "Frustum Line Width", initial_value=2.0, step=0.01, min=0.0, max=20.0
    )

    @gui_line_width.on_update
    def _(_) -> None:
        for cam in camera_frustums:
            cam.line_width = gui_line_width.value

    gui_frustum_scale = server.gui.add_slider(
        "Frustum Scale", initial_value=0.3, step=0.001, min=0.01, max=20.0
    )

    @gui_frustum_scale.on_update
    def _(_) -> None:
        for cam in camera_frustums:
            cam.scale = gui_frustum_scale.value

    gui_frustum_ours_color = server.gui.add_rgb(
        "Frustum RGB (ours)", initial_value=(255, 127, 14)
    )
    @gui_frustum_ours_color.on_update
    def _(_) -> None:
        for cam in camera_frustums:
            cam.color = gui_frustum_ours_color.value
            
    gui_confidence_threshold = server.gui.add_number(
            "Point Confidence >=", initial_value=3.5, min=0.0, max=20.0
        )
    gui_confidence_threshold.on_update(set_stale)

    gui_point_size = server.gui.add_slider(
        "Point Size", initial_value=0.01, step=0.0001, min=0.001, max=0.05
    )
    gui_point_white = server.gui.add_slider(
        "Point White", initial_value=0.0, step=0.0001, min=0.0, max=1.0
    )
    gui_point_white.on_update(set_stale)

    @gui_point_size.on_update
    def _(_) -> None:
        for pc in pointcloud_handles:
            pc.point_size = gui_point_size.value

    gui_point_shape = server.gui.add_dropdown(
        "Point Shape", options=("circle", "square"), initial_value="circle"
    )

    @gui_point_shape.on_update
    def _(_) -> None:
        for pc in pointcloud_handles:
            pc.point_ball_norm = (
                2.0 if gui_point_shape.value == "circle" else np.inf
            )    



    # get rotation matrix of 180 degrees around x axis
    rot_180 = np.eye(3)
    rot_180[1, 1] = -1
    rot_180[2, 2] = -1

    # Add GUI elements.
    timing_handle = server.gui.add_number("Time (ms)", 0.01, disabled=True)

    pointcloud_handles = []
    cam_handles = []
    camera_frustums = [] #list[viser.CameraFrustumHandle]()
    gt_cam_handles = []
    control_distance_measurement = None
    start_time = time.time()
    try:
        while True:
            time.sleep(0.1)
            timing_handle.value = (time.time() - start_time) 


            with server.atomic():
                for img_idx, img_name in enumerate(sorted(world_env.keys())):
                    if img_name == 'non_img_specific':
                        continue
                    # Visualize the pointcloud of environment
                    pts3d = world_env[img_name]['pts3d']
                    if pts3d.ndim == 3:
                        pts3d = pts3d.reshape(-1, 3)
                    # points = pts3d[world_env[img_name]['msk'].flatten()]
                    # colors = world_env[img_name]['rgbimg'][world_env[img_name]['msk']].reshape(-1, 3)
                    # no masking
                    points = pts3d
                    colors =world_env[img_name]['rgbimg'].reshape(-1, 3)

                    points = points @ rot_180

                    # Filter out points with confidence < 3.5.
                    mask = world_env[img_name]['conf'].flatten() >= gui_confidence_threshold.value
                    points_filtered = points[mask]
                    colors_filtered = colors[mask]

                    pointcloud_handle = server.scene.add_point_cloud(
                        f"/ours/pointcloud_{img_name}",
                        points=points_filtered,
                        point_size=gui_point_size.value,
                        point_shape=gui_point_shape.value,
                        colors=(1.0 - gui_point_white.value) * colors_filtered
                            + gui_point_white.value,
                    )
                    pointcloud_handles.append(pointcloud_handle)

                    # pc_handle = server.scene.add_point_cloud(
                    #     f"/pts3d_{img_name}",
                    #     points=points,
                    #     colors=colors,
                    #     point_size=0.05,
                    #     point_shape='circle'
                    # )
                    # pointcloud_handles.append(pc_handle)

                    # Visualize the camera
                    camera = world_env[img_name]['cam2world'].copy()
                    camera[:3, :3] = rot_180 @ camera[:3, :3] 
                    camera[:3, 3] = camera[:3, 3] @ rot_180
                    
                    # rotation matrix to quaternion
                    quat = R.from_matrix(camera[:3, :3]).as_quat()
                    # xyzw to wxyz
                    quat = np.concatenate([quat[3:], quat[:3]])
                    # translation vector
                    trans = camera[:3, 3]

                    # add camera frustum
                    rgbimg = world_env[img_name]['rgbimg']
                    K = world_env[img_name]['intrinsic']
                    # fov_rad = 2 * np.arctan(intrinsics_K[0, 2] / intrinsics_K[0, 0])DA
                    assert K.shape == (3, 3)
                    vfov_rad = 2 * np.arctan(K[1, 2] / K[1, 1])
                    aspect = rgbimg.shape[1] / rgbimg.shape[0]

                    camera_frustm = server.scene.add_camera_frustum(
                        f"/ours/{img_name}",
                        vfov_rad,
                        aspect,
                        scale=gui_frustum_scale.value,
                        line_width=gui_line_width.value,
                        color=gui_frustum_ours_color.value,
                        wxyz=quat,
                        position=trans,
                        image=rgbimg,
                    )
                    camera_frustums.append(camera_frustm)

                    # # add camera
                    # cam_handle = server.scene.add_frame(
                    #     f"/cam_{img_name}",
                    #     wxyz=quat,
                    #     position=trans,
                    #     show_axes=True,
                    #     axes_length=0.5,
                    #     axes_radius=0.04,
                    # )
                    # cam_handles.append(cam_handle)

                if smplx_vertices_dict is not None:
                    for img_idx, img_name in enumerate(sorted(smplx_vertices_dict.keys())):
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
                
                if gt_cameras is not None:
                    for img_name in gt_cameras.keys():
                        # Visualize the gt camera
                        # camera = gt_cameras[img_name]
                        # cam2world_Rt_homo = camera['cam2world_4by4'].copy()
                        cam2world_Rt_homo = gt_cameras[img_name]

                        cam2world_R = rot_180 @ cam2world_Rt_homo[:3, :3]
                        cam2world_t = cam2world_Rt_homo[:3, 3] @ rot_180

                        # rotation matrix to quaternion
                        quat = R.from_matrix(cam2world_R).as_quat()
                        # xyzw to wxyz
                        quat = np.concatenate([quat[3:], quat[:3]])
                        # translation vector
                        trans = cam2world_t   

                        # add camera
                        gt_cam_handle = server.scene.add_frame(
                            f"/gt_cam_{img_name}",
                            wxyz=quat,
                            position=trans,
                        )
                        gt_cam_handles.append(gt_cam_handle)

                # add transform controls, initialize the location with the first two cameras
                if control_distance_measurement is None:
                    control0 = server.scene.add_transform_controls(
                        "/controls/0",
                        position=camera_frustums[0].position,
                        scale=0.5
                    )
                    control1 = server.scene.add_transform_controls(
                        "/controls/1",
                        position=camera_frustums[1].position,
                        scale=0.5
                    )
                    distance_text = server.gui.add_text("Distance", initial_value="Distance: 0")
                    control_distance_measurement = True

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

    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
    tyro.cli(main)
