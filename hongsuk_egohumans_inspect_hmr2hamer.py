import pickle
import numpy as np
import torch
import smplx
import trimesh
import cv2
import os
import time
import viser
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D

from hongsuk_joint_names import COCO_WHOLEBODY_KEYPOINTS, ORIGINAL_SMPLX_JOINT_NAMES

coco_main_body_end_joint_idx = COCO_WHOLEBODY_KEYPOINTS.index('right_heel') 
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

def draw_2d_skeleton(img, keypoints, skeleton, color=(0, 255, 0), thickness=2):
    for edge in skeleton:
        pt1 = tuple(map(int, keypoints[edge[0]]))
        pt2 = tuple(map(int, keypoints[edge[1]]))
        cv2.line(img, pt1, pt2, color, thickness)

def visualize_3d_skeleton(joints_3d, skeleton, save_path):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot coordinate axes
    origin = np.zeros(3)
    axis_length = 0.3  # Adjust this value to change axes length
    
    # X-axis in red
    ax.quiver(origin[0], origin[1], origin[2], 
              axis_length, 0, 0, color='red')
    # Y-axis in green  
    ax.quiver(origin[0], origin[1], origin[2],
              0, axis_length, 0, color='green')
    # Z-axis in blue
    ax.quiver(origin[0], origin[1], origin[2],
              0, 0, axis_length, color='blue')
    
    # Label the axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y') 
    ax.set_zlabel('Z')

    # Flip the z axis
    joints_3d[:, 2] = -joints_3d[:, 2]
    
    # Plot joints
    ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], c='r', marker='o')
    
    # Plot skeleton
    for edge in skeleton:
        xs = [joints_3d[edge[0], 0], joints_3d[edge[1], 0]]
        ys = [joints_3d[edge[0], 1], joints_3d[edge[1], 1]]
        zs = [joints_3d[edge[0], 2], joints_3d[edge[1], 2]]
        ax.plot(xs, ys, zs, c='b')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1,1,1])
    plt.savefig(save_path)
    plt.close()

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

def main():
    smplx_data_dir = '/scratch/one_month/current/lmueller/egohuman/camera_ready/01_tagging/001_tagging/processed_data/humanwithhand/cam01/__hmr2_hamer_smplx.pkl'
    with open(smplx_data_dir, 'rb') as f:
        smplx_data = pickle.load(f)

    frame_key = '00001'
    print(smplx_data[frame_key].keys())

    params = smplx_data[frame_key]['params'] 
    # keys: 'body_pose', 'betas', 'global_orient', 'right_hand_pose', 'left_hand_pose', 'transl'
    additional_params = smplx_data[frame_key]['additional_params']
    # keys: 'hmr2_pred_keypoints_2d_hmr2', 'hamer_pred_keypoints_2d_hamer', 'hmr2_scaled_focal_length', 'hmr2_pred_cam_t_full', 'hmr2_img_size', 'hmr2_box_size', 'hmr2_box_center'
    print(params.keys())
    hmr2_scaled_focal_length = additional_params['hmr2_scaled_focal_length']
    hmr2_pred_cam_t_full = additional_params['hmr2_pred_cam_t_full']
    hmr2_img_size = additional_params['hmr2_img_size']
    # hmr2_box_size = additional_params['hmr2_box_size']
    # hmr2_box_center = additional_params['hmr2_box_center']

    num_persons = params['body_pose'].shape[0]
    # define the smplx layer
    start_time = time.time()
    smplx_layer = smplx.create(
        model_path = '/home/hongsuk/projects/egoexo/essentials/body_models',
        model_type = 'smplx',
        gender = 'neutral',
        use_pca = False,
        num_pca_comps = 45,
        flat_hand_mean = True,
        use_face_contour = True,
        num_betas = 10,
        batch_size = num_persons,
    )
    smplx_layer = smplx_layer.to('cuda')
    end_time = time.time()
    print(f"Time taken to define smplx layer: {end_time - start_time:.2f} seconds")
    import pdb; pdb.set_trace()

    body_pose = params['body_pose'].reshape(num_persons, -1)
    global_orient = params['global_orient'].reshape(num_persons, -1)
    betas = params['betas'].reshape(num_persons, -1)
    left_hand_pose = params['left_hand_pose'].reshape(num_persons, -1)
    right_hand_pose = params['right_hand_pose'].reshape(num_persons, -1)
    transl = params['transl'].reshape(num_persons, -1)

    # smpl-x forward path
    body = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=transl)

    # vertices_list = body.vertices # (N, 10475, 3)
    # joints_list = body.joints # (N, 144, 3)
    # # save the vertices as obj file
    # for i in range(num_persons):
    #     vertices = vertices_list[i].detach().cpu().numpy()

    #     mesh = trimesh.Trimesh(vertices, smplx_layer.faces)
    #     mesh.visual.face_colors = [255, 0, 0, 255*0.5] ## note the colors are bgr
    #     mesh.export(f'{frame_key}_smplx_{i}.obj')

    # load the image 
    img_path = '/scratch/partial_datasets/bedlam/egohumans/extracted/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready/01_tagging/001_tagging/exo/cam01/images/'
    img_name = f'{frame_key}.jpg'
    img = cv2.imread(os.path.join(img_path, img_name))

    # load the ViTkeypoints
    vit_data_dir = '/scratch/one_month/current/lmueller/egohuman/camera_ready/01_tagging/001_tagging/processed_data/humanwithhand/cam01/__vitpose.pkl'
    with open(vit_data_dir, 'rb') as f:
        vit_data = pickle.load(f)
    vitpose_2d_keypoints = vit_data[frame_key] # (N, 133, 3)

    
    # Draw 2D skeleton from ViTPose
    for i in range(num_persons):
        joints = vitpose_2d_keypoints[i]
        joints = joints[:coco_main_body_end_joint_idx + 1, :2]  # Only main body joints
        draw_2d_skeleton(img, joints, COCO_MAIN_BODY_SKELETON)
        
        # Draw joints on top of skeleton
        for j in range(joints.shape[0]):
            cv2.circle(img, (int(joints[j, 0]), int(joints[j, 1])), 3, (0, 0, 255), -1)
            cv2.putText(img, COCO_WHOLEBODY_KEYPOINTS[j], (int(joints[j, 0]), int(joints[j, 1]) - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Save the 2D visualization
    save_path = os.path.join('./vis_keypoints', f'{frame_key}_vitpose.jpg')
    cv2.imwrite(save_path, img)

    # Extract main body joints and visualize 3D skeleton from SMPL-X
    smplx_joints = body.joints
    smplx_coco_main_body_joints = smplx_joints[:, smplx_main_body_joint_idx, :].detach().cpu().numpy()
    
    # img = cv2.imread(os.path.join(img_path, img_name))
    # for i in range(num_persons):
    #     if i > 0:
    #         break
    #     # vis_3d_save_path = os.path.join('./vis_keypoints', f'{frame_key}_person{i}_3d_skeleton.png')
    #     # visualize_3d_skeleton(smplx_coco_main_body_joints[i], COCO_MAIN_BODY_SKELETON, vis_3d_save_path)
    #     joints = (smplx_coco_main_body_joints[i] * 500).astype(np.int32)[:, :2]
    #     # TEMP
    #     joints -= joints.min() 
    #     joints += np.array([[500, 500]])
    #     draw_2d_skeleton(img, joints, COCO_MAIN_BODY_SKELETON)

    #     # Draw joints on top of skeleton
    #     for j in range(joints.shape[0]):
    #         cv2.circle(img, (int(joints[j, 0]), int(joints[j, 1])), 3, (0, 0, 255), -1)
    #         cv2.putText(img, COCO_WHOLEBODY_KEYPOINTS[j], (int(joints[j, 0]), int(joints[j, 1]) - 2), 
    #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    # # Save the 2D visualization
    # save_path = os.path.join('./vis_keypoints', f'{frame_key}_smplx.jpg')
    # cv2.imwrite(save_path, img)

    vertices_list = body.vertices # (N, 10475, 3)
    joints_list = body.joints # (N, 144, 3)
    # Calculate focal length from field of view (60 degrees) and image size
    fov = 60  # degrees
    fov_rad = np.deg2rad(fov)
    focal_length = hmr2_img_size[0][1] / (2 * np.tan(fov_rad / 2))  # Using width of image
    print("Focal length: ", focal_length)
    # focal_length = 1400
    princpt = np.array([hmr2_img_size[0][1] / 2, hmr2_img_size[0][0] / 2])
    
    new_vertices_list = []
    init_trans_list = []
    for i in range(num_persons):
        init_trans = estimate_initial_trans(smplx_coco_main_body_joints[i], vitpose_2d_keypoints[i], focal_length, princpt, COCO_MAIN_BODY_SKELETON)
        print(f"person {i}: {init_trans}")

        vertices = vertices_list[i].detach().cpu().numpy()
        joints = joints_list[i].detach().cpu().numpy()

        vertices = vertices - joints[0:1] + init_trans
        new_vertices_list.append(vertices)
        init_trans_list.append(init_trans)
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
    for person_idx, vertices in enumerate(new_vertices_list):
        vertices = vertices @ rot_180       
        server.scene.add_mesh_simple(
            f"/{frame_key}_smplx_person{person_idx}/mesh",
            vertices=vertices,
            faces=smplx_layer.faces,
            flat_shading=False,
            wireframe=False,
            color=get_color(person_idx),
        )


    # add transform controls, initialize the location with the first two cameras
    control0 = server.scene.add_transform_controls(
        "/controls/0",
        position=np.array([0, 0, 0]),
        scale=0.4,
    )
    init_trans = init_trans_list[0] @ rot_180
    control1 = server.scene.add_transform_controls(
        "/controls/1",
        position=init_trans,
        scale=0.4,
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
    main()