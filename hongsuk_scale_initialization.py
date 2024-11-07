


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

def init_human_params(smplx_layer, multiview_multiple_human_cam_pred, multiview_multiperson_pose2d, focal_length, princpt, device = 'cuda', get_vertices=False):
    # import smplx
    # smplx_layer = smplx.create(
    #     model_path = '/home/hongsuk/projects/egoexo/essentials/body_models',
    #     model_type = 'smplx',
    #     gender = 'neutral',
    #     use_pca = False,
    #     num_pca_comps = 45,
    #     flat_hand_mean = True,
    #     use_face_contour = True,
    #     num_betas = 10,
    #     batch_size = 1,
    # )
    # multiview_multiple_human_cam_pred: Dict[camera_name -> Dict[human_name -> 'pose2d', 'bbox', 'params' Dicts]]
    # multiview_multiperson_pose2d: Dict[human_name -> Dict[cam_name -> (J, 2+1)]] torch tensor
    # focal_length: scalar, princpt: (2,), device: str
    # focal_length and princpt are from Dust3r; usually for Duster image size of (288,512),       
    # focal_length = 314
    # init_princpt = [256., 144.] 

    # Initialize Stage 1: Get the 3D root translation of all humans from all cameras
    # Decode the smplx mesh and get the 3D bone lengths / compare them with the bone lengths from the vitpose 2D bone lengths
    camera_names = sorted(list(multiview_multiple_human_cam_pred.keys()))
    first_cam = camera_names[0]
    first_cam_human_name_counts = {human_name: 0 for human_name in multiview_multiple_human_cam_pred[first_cam].keys()}
    missing_human_names_in_first_cam = defaultdict(list)
    multiview_multiperson_init_trans = defaultdict(dict) # Dict[human_name -> Dict[cam_name -> (3)]]
    for cam_name in camera_names:
        for human_name in multiview_multiple_human_cam_pred[cam_name].keys():
            params = multiview_multiple_human_cam_pred[cam_name][human_name]['params']
            body_pose = params['body_pose'].reshape(1, -1).to(device)
            global_orient = params['global_orient'].reshape(1, -1).to(device)
            betas = params['betas'].reshape(1, -1).to(device)
            left_hand_pose = params['left_hand_pose'].reshape(1, -1).to(device)
            right_hand_pose = params['right_hand_pose'].reshape(1, -1).to(device)
            transl = params['transl'].reshape(1, -1).to(device)
            
            smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=transl)

            # Extract main body joints and visualize 3D skeleton from SMPL-X
            smplx_joints = smplx_output['joints']
            # Save the root joint (pelvis) translation for later compensation
            params['org_cam_root_transl'] = smplx_joints[0, 0,:3].detach().cpu().numpy()

            smplx_coco_main_body_joints = smplx_joints[0, smplx_main_body_joint_idx, :].detach().cpu().numpy()
            vitpose_2d_keypoints = multiview_multiperson_pose2d[human_name][cam_name][coco_main_body_joint_idx].cpu().numpy() # (J, 2+1)
            init_trans = estimate_initial_trans(smplx_coco_main_body_joints, vitpose_2d_keypoints, focal_length, princpt, COCO_MAIN_BODY_SKELETON)
            # How to use init_trans?
            # vertices = vertices - joints[0:1] + init_trans # !ALWAYS! Fuck the params['transl']
            if human_name in first_cam_human_name_counts.keys():
                first_cam_human_name_counts[human_name] += 1
            else:
                missing_human_names_in_first_cam[human_name].append(cam_name)
            multiview_multiperson_init_trans[human_name][cam_name] = init_trans

    # main human is the one that is detected in the first camera and has the most detections across all cameras
    main_human_name = None
    max_count = 0
    for human_name, count in first_cam_human_name_counts.items():
        if count == len(camera_names):
            main_human_name = human_name
            max_count = len(camera_names)
            break
        elif count > max_count:
            max_count = count
            main_human_name = human_name
    if max_count != len(camera_names):
        print(f"Warning: {main_human_name} is the most detected main human but not detected in all cameras")
    
    # Initialize Stage 2: Get the initial camera poses with respect to the first camera
    global_orient_first_cam = multiview_multiple_human_cam_pred[first_cam][main_human_name]['params']['global_orient'][0].cpu().numpy()
    # axis angle to rotation matrix
    global_orient_first_cam = R.from_rotvec(global_orient_first_cam).as_matrix().astype(np.float32)
    init_trans_first_cam = multiview_multiperson_init_trans[main_human_name][first_cam]

    # First camera (world coordinate) pose
    world_T_first = np.eye(4, dtype=np.float32)  # Identity rotation and zero translation

    # Calculate other camera poses relative to world (first camera)
    cam_poses = {first_cam: world_T_first}  # Store all camera poses
    for cam_name in multiview_multiperson_init_trans[main_human_name].keys():
        if cam_name == first_cam:
            continue
        
        # Get human orientation and position in other camera
        global_orient_other_cam = multiview_multiple_human_cam_pred[cam_name][main_human_name]['params']['global_orient'][0].cpu().numpy()
        global_orient_other_cam = R.from_rotvec(global_orient_other_cam).as_matrix().astype(np.float32)
        init_trans_other_cam = multiview_multiperson_init_trans[main_human_name][cam_name]

        # The human's orientation should be the same in world coordinates
        # Therefore: R_other @ global_orient_other = R_first @ global_orient_first
        # Solve for R_other: R_other = (R_first @ global_orient_first) @ global_orient_other.T
        R_other = global_orient_first_cam @ global_orient_other_cam.T

        # For translation: The human position in world coordinates should be the same when viewed from any camera
        # world_p = R_first @ p_first + t_first = R_other @ p_other + t_other
        # Since R_first = I and t_first = 0:
        # p_first = R_other @ p_other + t_other
        # Solve for t_other: t_other = p_first - R_other @ p_other
        t_other = init_trans_first_cam - R_other @ init_trans_other_cam

        # Create 4x4 transformation matrix
        T_other = np.eye(4, dtype=np.float32)
        T_other[:3, :3] = R_other
        T_other[:3, 3] = t_other

        cam_poses[cam_name] = T_other

    # Visualize the camera poses (cam to world (first cam))
    # visualize_cameras(cam_poses)

    # Now cam_poses contains all camera poses in world coordinates
    # The poses can be used to initialize the scene

    # Organize the data for optimization
    # Get the first cam human parameters with the initial translation
    first_cam_human_params = {}
    for human_name in multiview_multiple_human_cam_pred[first_cam].keys():
        first_cam_human_params[human_name] = multiview_multiple_human_cam_pred[first_cam][human_name]['params']
        first_cam_human_params[human_name]['root_transl'] = torch.from_numpy(multiview_multiperson_init_trans[human_name][first_cam]).reshape(1, -1).to(device)

    # Initialize Stage 3: If the first camera (world coordinate frame) has missing person,
    # move other camera view's human to the first camera view's human's location
    for missing_human_name in missing_human_names_in_first_cam:
        missing_human_exist_cam_idx = 0
        other_cam_name = missing_human_names_in_first_cam[missing_human_name][missing_human_exist_cam_idx]
        while other_cam_name not in cam_poses.keys():
            missing_human_exist_cam_idx += 1
            if missing_human_exist_cam_idx == len(missing_human_names_in_first_cam[missing_human_name]):
                print(f"Warning: {missing_human_name} cannot be handled because it can't transform to the first camera coordinate frame")
                continue
            other_cam_name = missing_human_names_in_first_cam[missing_human_name][missing_human_exist_cam_idx]
        missing_human_params_in_other_cam = multiview_multiple_human_cam_pred[other_cam_name][missing_human_name]['params']
        # keys: 'body_pose', 'betas', 'global_orient', 'right_hand_pose', 'left_hand_pose', 'transl'
        # transform the missing_human_params_in_other_cam to the first camera coordinate frame
        other_cam_to_first_cam_transformation = cam_poses[other_cam_name] # (4,4)
        missing_human_params_in_other_cam_global_orient = missing_human_params_in_other_cam['global_orient'][0].cpu().numpy() # (3,)
        missing_human_params_in_other_cam_global_orient = R.from_rotvec(missing_human_params_in_other_cam_global_orient).as_matrix().astype(np.float32) # (3,3)
        missing_human_params_in_other_cam_global_orient = other_cam_to_first_cam_transformation[:3, :3] @ missing_human_params_in_other_cam_global_orient # (3,3)
        missing_human_params_in_other_cam['global_orient'] = torch.from_numpy(R.from_matrix(missing_human_params_in_other_cam_global_orient).as_rotvec().astype(np.float32)).to(device) # (3,)

        missing_human_init_trans_in_other_cam = multiview_multiperson_init_trans[missing_human_name][other_cam_name]
        missing_human_init_trans_in_first_cam = other_cam_to_first_cam_transformation[:3, :3] @ missing_human_init_trans_in_other_cam + other_cam_to_first_cam_transformation[:3, 3]
        # compenstate rotation (translation from origin to root joint was not cancled)
        root_transl_compensator = other_cam_to_first_cam_transformation[:3, :3] @ missing_human_params_in_other_cam['org_cam_root_transl'] 
        missing_human_init_trans_in_first_cam = missing_human_init_trans_in_first_cam + root_transl_compensator
        #
        missing_human_params_in_other_cam['root_transl'] = torch.from_numpy(missing_human_init_trans_in_first_cam).reshape(1, -1).to(device)

        first_cam_human_params[missing_human_name] = missing_human_params_in_other_cam

    # Visualize the first cam human parameters with the camera poses
    # decode the human parameters to 3D vertices and visualize
    if get_vertices:
        first_cam_human_vertices = {}
        for human_name, human_params in first_cam_human_params.items():
            body_pose = human_params['body_pose'].reshape(1, -1).to(device)
            global_orient = human_params['global_orient'].reshape(1, -1).to(device)
            betas = human_params['betas'].reshape(1, -1).to(device)
            left_hand_pose = human_params['left_hand_pose'].reshape(1, -1).to(device)
            right_hand_pose = human_params['right_hand_pose'].reshape(1, -1).to(device)
            transl = human_params['transl'].reshape(1, -1).to(device)

            smplx_output = smplx_layer(body_pose=body_pose, betas=betas, global_orient=global_orient, left_hand_pose=left_hand_pose, right_hand_pose=right_hand_pose, transl=transl)

            vertices = smplx_output.vertices[0].detach().cpu().numpy()
            joints = smplx_output.joints[0].detach().cpu().numpy()
            vertices = vertices - joints[0:1] + human_params['root_transl'].cpu().numpy()
            first_cam_human_vertices[human_name] = vertices
            # visualize_cameras_and_human(cam_poses, human_vertices=first_cam_human_vertices, smplx_faces=smplx_layer.faces)
    else:
        first_cam_human_vertices = None

    optim_target_dict = {} # human_name: str -> Dict[param_name: str -> nn.Parameter]
    for human_name, human_params in first_cam_human_params.items():
        optim_target_dict[human_name] = {}
        
        # Convert human parameters to nn.Parameters for optimization
        optim_target_dict[human_name]['body_pose'] = nn.Parameter(human_params['body_pose'].float().to(device))  # (1, 63)
        optim_target_dict[human_name]['global_orient'] = nn.Parameter(human_params['global_orient'].float().to(device))  # (1, 3) 
        optim_target_dict[human_name]['betas'] = nn.Parameter(human_params['betas'].float().to(device))  # (1, 10)
        optim_target_dict[human_name]['left_hand_pose'] = nn.Parameter(human_params['left_hand_pose'].float().to(device))  # (1, 45)
        optim_target_dict[human_name]['right_hand_pose'] = nn.Parameter(human_params['right_hand_pose'].float().to(device))  # (1, 45)
        optim_target_dict[human_name]['root_transl'] = nn.Parameter(human_params['root_transl'].float().to(device)) # (1, 3)

        # TEMP
        # make relative_rotvec, shape, expression not require grad
        optim_target_dict[human_name]['body_pose'].requires_grad = False
        # optim_target_dict[human_name]['global_orient'].requires_grad = False
        optim_target_dict[human_name]['betas'].requires_grad = False
        optim_target_dict[human_name]['left_hand_pose'].requires_grad = False
        optim_target_dict[human_name]['right_hand_pose'].requires_grad = False

    return optim_target_dict, cam_poses, first_cam_human_vertices



def main():

    """ Initialize Scale """
    human_params, human_inited_cam_poses, first_cam_human_vertices = \
        init_human_params(smplx_layer, multiview_multiple_human_cam_pred, multiview_multiperson_poses2d, init_focal_length, init_princpt, device, get_vertices=vis) # dict of human parameters

    # Initialize the scale factor between the dust3r cameras and the human_inited_cam_poses
    # Perform Procrustes alignment
    human_inited_cam_locations = []
    dust3r_cam_locations = []
    for cam_name in human_inited_cam_poses.keys():
        human_inited_cam_locations.append(human_inited_cam_poses[cam_name][:3, 3])
        dust3r_cam_locations.append(im_poses[cam_names.index(cam_name)][:3, 3].cpu().numpy())
    human_inited_cam_locations = np.array(human_inited_cam_locations)
    dust3r_cam_locations = np.array(dust3r_cam_locations)

    if len(human_inited_cam_locations) > 2:
        scale, _, _ = procrustes_align(human_inited_cam_locations, dust3r_cam_locations)
    elif len(human_inited_cam_locations) == 2:
        # get the ratio between the two distances
        dist_ratio = np.linalg.norm(human_inited_cam_locations[0] - human_inited_cam_locations[1]) / np.linalg.norm(dust3r_cam_locations[0] - dust3r_cam_locations[1])
        scale = dist_ratio
    else:
        print("Not enough camera locations to perform Procrustes alignment or distance ratio calculation")
        scale = 100.0