import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.spatial
from scipy.optimize import linear_sum_assignment
import os
import json
import cv2
import glob
import PIL
from PIL import Image, ImageOps
import pickle
from collections import defaultdict
from dust3r.utils.image import load_images as dust3r_load_images
from multihmr.utils import normalize_rgb, get_focalLength_from_fieldOfView
from multihmr.utils import get_smplx_joint_names

from hongsuk_joint_names import COCO_WHOLEBODY_KEYPOINTS, SMPLX_JOINT_NAMES

class EgoHumansDataset(Dataset):
    def __init__(self, data_root, dust3r_raw_output_dir=None, dust3r_ga_output_dir=None, vitpose_hmr2_hamer_output_dir=None, multihmr_output_path=None, split='train', subsample_rate=10, cam_names=None, num_of_cams=None, selected_big_seq_list=[]):
        """
        Args:
            data_root (str): Root directory of the dataset
            dust3r_raw_output_dir (str): Directory to the dust3r raw outputs
            dust3r_ga_output_dir (str): Directory to the dust3r global alignment outputs
            multihmr_output_path (str): Path to the multihmr output
            split (str): 'train', 'val', or 'test'
        """
        self.data_root = data_root 
        self.split = split
        self.dust3r_image_size = 512
        self.multihmr_image_size = 896
        self.egohumans_image_size_tuple = (2160, 3840) # (height, width)
        self.subsample_rate = subsample_rate

        # choose camera names
        if cam_names is None:
            self.camera_names = None
            self.num_of_cams = num_of_cams
        else:
            self.camera_names = cam_names
            self.num_of_cams = len(cam_names)

        # choose only a few sequence for testing else all sequences
        # 048_badminton to 051_badminton
        start_idx = 41#55#51
        end_idx = 61#58#54
        self.selected_small_seq_name_list = [f'{i:03d}_badminton' for i in range(start_idx, end_idx+1)]

        # self.selected_small_seq_name_list = [] #'001_badminton']  # ex) ['001_tagging', '002_tagging']
        self.selected_big_seq_list = selected_big_seq_list # ex) ['01_tagging', '02_lego']
        # big sequence name dictionary
        self.big_seq_name_dict = {
            'tagging': '01_tagging',
            'legoassemble': '02_lego',
            'fencing': '03_fencing',
            'basketball': '04_basketball',
            'volleyball': '05_volleyball',  
            'badminton': '06_badminton',
            'tennis': '07_tennis',
        }

        # Load dust3r network output
        self.dust3r_raw_output_dir = dust3r_raw_output_dir
        self.dust3r_raw_outputs = {}  # {small_seq_name: {frame: dust3r_raw_output_path}}
        if self.dust3r_raw_output_dir is not None:
            # Get all pkl files in the directory
            dust3r_raw_files = sorted(glob.glob(os.path.join(self.dust3r_raw_output_dir, '*.pkl')))
            
            for file_path in dust3r_raw_files:
                # Get filename without extension; ex) 001_badminton_0.pkl
                filename = os.path.basename(file_path).replace('.pkl', '')
                if self.selected_small_seq_name_list != [] and '_'.join(filename.split('_')[:-1]) not in self.selected_small_seq_name_list:
                    continue
                
                # Split filename into sequence and frame
                seq_parts = filename.split('_')
                small_seq_name = '_'.join(seq_parts[:2])  # e.g. '001_badminton'
                frame = int(seq_parts[2])  # e.g. 0, 10, etc.
                
                # Initialize dictionary for this sequence if not exists
                if small_seq_name not in self.dust3r_raw_outputs:
                    self.dust3r_raw_outputs[small_seq_name] = {}
                
                # Store the path
                self.dust3r_raw_outputs[small_seq_name][frame] = file_path
                

            """ > Structure + Humans + Cameras optimization - Hongsuk (optional): """
            # Load dust3r global alignment output
            self.dust3r_ga_output_dir = dust3r_ga_output_dir
            self.dust3r_ga_outputs = {} # {small_seq_name: {frame: dust3r_ga_output_path}}
            if self.dust3r_ga_output_dir is not None:
                for small_seq_name in self.dust3r_raw_outputs.keys():
                    for frame in self.dust3r_raw_outputs[small_seq_name].keys():
                        dust3r_ga_output_path = os.path.join(self.dust3r_ga_output_dir, f'{small_seq_name}_{frame}.pkl')
                        if not os.path.exists(dust3r_ga_output_path):
                            print(f'Error: {dust3r_ga_output_path} does not exist')
                            raise ValueError(f'{dust3r_ga_output_path} does not exist')

                        # initialize dictionary for this sequence if not exists
                        if small_seq_name not in self.dust3r_ga_outputs:
                            self.dust3r_ga_outputs[small_seq_name] = {}

                        # Store the path
                        self.dust3r_ga_outputs[small_seq_name][frame] = dust3r_ga_output_path

        # Setup the VitPose / HMR2 / HAMER outputs 
        self.vitpose_hmr2_hamer_output_dir = '/scratch/one_month/current/lmueller/egohuman/camera_ready' #vitpose_hmr2_hamer_output_dir 


        # Load multihmr output
        self.multihmr_output_path = multihmr_output_path
        if self.dust3r_raw_output_dir is not None and self.dust3r_ga_output_dir is not None and self.multihmr_output_path is not None:
            multihmr_output = pickle.load(open(self.multihmr_output_path, 'rb'))
            self.multihmr_output = {} # {output_name: output}
            no_multihmr_output_names = []
            for output_name in self.dust3r_raw_outputs.keys():
                if output_name not in multihmr_output.keys():
                    no_multihmr_output_names.append(output_name)
                else:
                    self.multihmr_output[output_name] = multihmr_output[output_name]
            if len(no_multihmr_output_names) > 0:
                print(f'Warning: {no_multihmr_output_names} does not have multihmr output')
        """ Structure + Humans + Cameras optimization - Hongsuk (optional) < """

        # Load dataset metadata and create sample list
        self._load_annot_paths_and_cameras()
        self.datalist = self._load_dataset_info()
        print(f'Successfully loaded {len(self.datalist)} multiview samples')


    def _load_annot_paths_and_cameras(self):
        if self.selected_big_seq_list != []:
            self.big_seq_list = [os.path.join(self.data_root, big_seq) for big_seq in self.selected_big_seq_list]
        else:
            self.big_seq_list = sorted(glob.glob(os.path.join(self.data_root, '*')))
        print(f"Big sequence list: {self.big_seq_list}")
        self.small_seq_list = []
        self.small_seq_annot_list = []
        self.cameras = {}
        for big_seq in self.big_seq_list:
            small_seq_list = sorted(glob.glob(os.path.join(big_seq, '*')))
            
            for small_seq in small_seq_list:
                small_seq_name = os.path.basename(small_seq)
                if self.selected_small_seq_name_list != [] and small_seq_name not in self.selected_small_seq_name_list:
                    continue
                try:
                    with open(os.path.join(small_seq, 'parsed_annot_hongsuk.pkl'), 'rb') as f:
                        annot = pickle.load(f)

                    self.cameras[small_seq_name] = annot['cameras']
                    # print(f'Successfully loaded annot for {small_seq}')
                    self.small_seq_list.append(small_seq_name)
                    self.small_seq_annot_list.append(os.path.join(small_seq, 'parsed_annot_hongsuk.pkl'))
                except:
                    print(f'Warning: loading annot for {small_seq} failed') # there is no smpl annot for this sequence
        print(f'Successfully loaded annot for {len(self.small_seq_list)} sequences')

        """ Data Structure
        {
        num_frames: 
        , cameras: {
            camera_name1: {
                'cam2world_R': 
                'cam2world_t': 
                'K': 
                'img_width': 
                'img_height': 
            },
            ...
        }
        ,frame_data: [{
            world_data: 
                {
                    'human_name': {
                        'global_orient': 
                        'transl': 
                        'betas': 
                        'body_pose': 
                    }, 
                    ....
                }
            ]
        }]
        , per_view_2d_annot: {
            camera_name1: {
                'pose2d_annot_path_list': [],
                'bbox_annot_path_list': [],
            },
            camera_name2: {
                'pose2d_annot_path_list': [],
                'bbox_annot_path_list': [],
            },
            ...
        }

        } # end of dict
        """


    def _load_dataset_info(self):
        """Load and organize dataset information"""
        per_frame_data_list = []

        for small_seq, small_seq_annot in zip(self.small_seq_list, self.small_seq_annot_list):
            with open(small_seq_annot, 'rb') as f:
                annot = pickle.load(f)
        
            num_frames = annot['num_frames']

            for frame in range(num_frames):
                if frame % self.subsample_rate != 0:
                    continue

                per_frame_data = {
                    'sequence': small_seq,
                    'frame': frame
                }

                # add world smpl params data
                per_frame_data['world_data'] = annot['frame_data'][frame]['world_data'] # dictionrary of human names and their smpl params

                # check whether the dust3r_raw_output_dir is given and the dust3r output exists; this is fore global alignment step of Dust3r
                if self.dust3r_raw_output_dir is not None:
                    if small_seq in self.dust3r_raw_outputs.keys() and frame in self.dust3r_raw_outputs[small_seq].keys():
                        dust3r_raw_output_path = self.dust3r_raw_outputs[small_seq][frame]
                        per_frame_data['dust3r_raw_output_path'] = dust3r_raw_output_path
                        if self.dust3r_ga_output_dir is not None:
                            if small_seq in self.dust3r_ga_outputs.keys() and frame in self.dust3r_ga_outputs[small_seq].keys():
                                dust3r_ga_output_path = self.dust3r_ga_outputs[small_seq][frame]
                                per_frame_data['dust3r_ga_output_path'] = dust3r_ga_output_path
                            else:
                                print(f'Error: {small_seq}_{frame} does not have dust3r ga output')
                                raise ValueError(f'{small_seq}_{frame} does not have dust3r ga output')
                    else:
                        print(f'Warning: {small_seq}_{frame} does not have dust3r raw output')
                        continue

                """ add camera data """
                per_frame_data['cameras'] = annot['cameras'] # dictionrary of camera names and their parameters
                
                if self.camera_names is not None:
                    # check whether the cameras.keys() are superset of the self.camera_names
                    # if not, skip this frame
                    selected_cameras = sorted([cam for cam in per_frame_data['cameras'].keys() if cam in self.camera_names ])
                    if len(selected_cameras) != len(self.camera_names):
                        # print(f'Warning: {small_seq} does not have all the cameras in self.camera_names; skipping this frame')
                        continue
                else:
                    if self.num_of_cams is None:
                        print('Warning: Both camera names and number of cameras are None... Using all available cameras')
                        selected_cameras = sorted(per_frame_data['cameras'].keys())
                    else:
                        available_cameras = sorted(per_frame_data['cameras'].keys())
                        if len(available_cameras) < self.num_of_cams:
                            selected_cameras = available_cameras
                        else:
                            if self.num_of_cams <= 2:
                                selected_cameras = available_cameras[:self.num_of_cams]
                            else:
                                # Uniform sampling for more than 2 cameras
                                indices = np.linspace(0, len(available_cameras)-1, self.num_of_cams, dtype=int)
                                selected_cameras = [available_cameras[i] for i in indices]

                # sanitize camera data
                per_frame_data['cameras'] = {}
                for cam in selected_cameras:
                    annot_cam = annot['cameras'][cam]
                    per_frame_data['cameras'][cam] = annot_cam 

                    # cam2world_R = annot_cam['cam2world_R']
                    # cam2world_t = annot_cam['cam2world_t']

                    # # make 4by4 transformation matrix and save
                    # cam2world_Rt = np.concatenate((cam2world_R, cam2world_t[:, None]), axis=1)
                    # cam2world_Rt_4by4 = np.concatenate((cam2world_Rt, np.array([[0, 0, 0, 1]])), axis=0)
                    # per_frame_data['cameras'][cam]['cam2world_4by4'] = cam2world_Rt_4by4
                    per_frame_data['cameras'][cam]['cam2world_4by4'] = annot_cam['cam2world']
                    
                    # # make intrinsic matrix and save 
                    # K = annot_cam['K'] # (8,)        fx = params[0]
                    # fx = K[0]
                    # fy = K[1]
                    # cx = K[2]
                    # cy = K[3]
                    # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    # per_frame_data['cameras'][cam]['K'] = K

                # add 2d pose and bbox annot data
                per_frame_data['annot_and_img_paths'] = {}
                for camera_name in selected_cameras:
                    pose2d_annot_path = annot['per_view_2d_annot'][camera_name]['pose2d_annot_path_list'][frame] # numpy pickle
                    if len(annot['per_view_2d_annot'][camera_name]['bbox_annot_path_list']) > 0:
                        bbox_annot_path = annot['per_view_2d_annot'][camera_name]['bbox_annot_path_list'][frame] # numpy pickle
                    else:
                        bbox_annot_path = None
                    img_path = pose2d_annot_path.replace('processed_data/poses2d', 'exo').replace('rgb', 'images').replace('npy', 'jpg')

                    # replace the local path to the data root
                    img_path = img_path.replace('/home/hongsuk/projects/egohumans/data', self.data_root)
                    pose2d_annot_path = pose2d_annot_path.replace('/home/hongsuk/projects/egohumans/data', self.data_root)
                    if bbox_annot_path is not None:
                        bbox_annot_path = bbox_annot_path.replace('/home/hongsuk/projects/egohumans/data', self.data_root)

                    per_frame_data['annot_and_img_paths'][camera_name] = {
                        'pose2d_annot_path': pose2d_annot_path,
                        'bbox_annot_path': bbox_annot_path,
                        'img_path': img_path
                    }
                    # do this in __getitem__
                    # pose2d_annot = np.load(pose2d_annot_path, allow_pickle=True)
                    # bbox_annot = np.load(bbox_annot_path, allow_pickle=True)

                per_frame_data_list.append(per_frame_data)                

        return per_frame_data_list
    
    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        return self.get_single_item(idx)
    
    def get_single_item(self, idx):
        sample = self.datalist[idx]
        seq = sample['sequence']
        frame = sample['frame']

        """ get camera parameters; GT parameters """
        cameras = sample['cameras'] 
        # Align cameras to first camera frame
        first_cam = sorted(cameras.keys())[0]
        first_cam_Rt_4by4 = cameras[first_cam]['cam2world_4by4'].copy()

        # make camera parameters homogeneous and invert
        first_cam_Rt_inv_4by4 = np.linalg.inv(first_cam_Rt_4by4)
        
        # Transform all cameras relative to first camera
        for cam in sorted(cameras.keys()):
            cam_Rt_4by4 = cameras[cam]['cam2world_4by4']
            new_cam_Rt_4by4 = first_cam_Rt_inv_4by4 @ cam_Rt_4by4
            cameras[cam]['cam2world_4by4'] = new_cam_Rt_4by4

        """ load 3d world annot; GT parameters """
        world_multiple_human_3d_annot = {}
        for human_name in sample['world_data'].keys():
            world_multiple_human_3d_annot[human_name] = sample['world_data'][human_name]

        """ align the world annot to the first camera frame 
        # but it's messy to get correct transformation...
        # align the world annot to the first camera frame
        for human_name, smplx_3d_params in world_multiple_human_3d_annot.items():
            transl, global_orient = smplx_3d_params['transl'], smplx_3d_params['global_orient']

            world_multiple_human_3d_annot[human_name]['transl'] = first_cam_Rt_inv_4by4[:3, :3] @ transl + first_cam_Rt_inv_4by4[:3, 3]
            # world_multiple_human_3d_annot[human_name]['global_orient'] = first_cam_Rt_inv_4by4[:3, :3] @ global_orient
            # Convert global_orient from axis-angle to rotation matrix
            global_orient_mat = cv2.Rodrigues(global_orient)[0]  # (3,3)
            # Multiply by first camera rotation
            aligned_global_orient_mat = first_cam_Rt_inv_4by4[:3, :3] @ global_orient_mat
            
            # Convert back to axis-angle representation
            aligned_global_orient, _ = cv2.Rodrigues(aligned_global_orient_mat)
            # Update the global orientation
            world_multiple_human_3d_annot[human_name]['global_orient'] = aligned_global_orient.reshape(-1)


        # TEMPORARY CODE FOR VISUALIZATION
        import smplx
        import matplotlib.pyplot as plt
        from collections import defaultdict
        device = 'cuda'
        # for visualization, transform the human parameters to each camera frame and project to 2D and draw 2D keypoints and save them
        smpl_model_dir = '/home/hongsuk/projects/egohumans/egohumans/external/cliff/common/../data'
        smpl_model = smplx.create(smpl_model_dir, "smpl").to(device)
        smpl_model = smpl_model.float()
        

        vis_data = defaultdict(dict)
        for human_name, world_data_human in world_multiple_human_3d_annot.items():

            # first decode the smplx parameters
            smpl_output = smpl_model(betas=torch.from_numpy(world_data_human['betas'])[None, :].to(device).float(),
                                        body_pose=torch.from_numpy(world_data_human['body_pose'])[None, :].to(device).float(),
                                        global_orient=torch.from_numpy(world_data_human['global_orient'])[None, :].to(device).float(),
                                        pose2rot=True,
                                        transl=torch.zeros((1, 3), device=device))
                                        # transl=torch.from_numpy(world_data_human['transl'])[None, :].to(device).float())

            # compenstate rotation (translation from origin to root joint was not cancled)
            smpl_joints = smpl_output.joints.detach().squeeze(0).cpu().numpy()
            root_joint_coord = smpl_joints[0:1, :3]
            smpl_trans = world_data_human['transl'].reshape(1, 3) - root_joint_coord + np.dot(first_cam_Rt_inv_4by4[:3, :3], root_joint_coord.transpose(1, 0)).transpose(1, 0)

            smpl_vertices = smpl_output.vertices.detach().squeeze(0).cpu().numpy()
            smpl_vertices += smpl_trans
            smpl_joints += smpl_trans

            for cam in sorted(cameras.keys()):
                cam2world_4by4 = cameras[cam]['cam2world_4by4']
                world2cam_4by4 = np.linalg.inv(cam2world_4by4)
                intrinsic = cameras[cam]['K']
                # convert the smpl_joitns in world coordinates to camera coordinates
                # smpl_joints are in world coordinates and they are numpy arrays
                points_cam = world2cam_4by4 @ np.concatenate((smpl_joints, np.ones((smpl_joints.shape[0], 1))), axis=1).T # (4, J)
                points_cam = points_cam[:3, :].T # (J, 3)

                points_img = vec_image_from_cam(intrinsic, points_cam) # (J, 2)

                vis_data[cam][human_name] = points_img

        # Draw keypoints on images and save them
        for cam in vis_data.keys():
            # Load image
            img_path = sample['annot_and_img_paths'][cam]['img_path']
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Draw keypoints for each human
            for human_name, keypoints in vis_data[cam].items():
                # Draw each keypoint as a circle
                for kp in keypoints:
                    x, y = int(kp[0]), int(kp[1])
                    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:  # Only draw if within image bounds
                        cv2.circle(img, (x, y), 3, (255,0,0), -1)
            
                # Save the annotated image
                save_dir = os.path.join('./vis_keypoints')
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f'{cam}_{human_name}.jpg')
                plt.imsave(save_path, img)
        """
        
        # filter camera names that exist in both self.camera_names and cameras
        # selected_cameras = sorted([cam for cam in cameras.keys() if cam in self.camera_names])
        selected_cameras = sorted(sample['annot_and_img_paths'].keys())

        """ load image for Dust3r network inferences """
        multiview_images = {}
        img_path_list = [sample['annot_and_img_paths'][cam]['img_path'] for cam in selected_cameras]
        dust3r_input_imgs = dust3r_load_images(img_path_list, size=self.dust3r_image_size, verbose=False)

        # squeeze the batch dimension for 'img' and 'true_shape'
        new_dust3r_input_imgs = []
        for dust3r_input_img in dust3r_input_imgs:
            new_dust3r_input_img = {}
            for key in dust3r_input_img.keys(): # dict_keys(['img', 'true_shape', 'idx', 'instance'])
                if key in ['img', 'true_shape']: #
                    new_dust3r_input_img[key] = dust3r_input_img[key][0]
                else:
                    new_dust3r_input_img[key] = dust3r_input_img[key]
            new_dust3r_input_imgs.append(new_dust3r_input_img)
        multiview_images = {cam: new_dust3r_input_imgs[i] for i, cam in enumerate(selected_cameras)}
        multiview_affine_transforms = {cam: get_dust3r_affine_transform(img_path, size=self.dust3r_image_size) for cam, img_path in zip(selected_cameras, img_path_list)}

        """ load image for MultiHMR inference """
        # let's use the first camera for now
        first_cam_img_path = sample['annot_and_img_paths'][first_cam]['img_path'].replace('/home/hongsuk/projects/egohumans/data', self.data_root)
        multihmr_first_cam_input_image, _, multihmr_first_cam_affine_matrix = get_multihmr_input_image(first_cam_img_path, self.multihmr_image_size)
        multihmr_first_cam_intrinsic = get_multihmr_camera_parameters(self.multihmr_image_size)

        """ get dust3r output; Predicted parameters """
        if self.dust3r_raw_output_dir is not None:
            dust3r_raw_output_path = sample['dust3r_raw_output_path']
            with open(dust3r_raw_output_path, 'rb') as f:
                dust3r_raw_outputs = pickle.load(f)
        
            dust3r_network_output = dust3r_raw_outputs['output']
            dust3r_raw_output_camera_names = dust3r_raw_outputs['img_names'][2]

            assert dust3r_raw_output_camera_names == selected_cameras, "The camera names in the dust3r raw output and the selected cameras do not match!"
            del dust3r_network_output['loss']

            if self.dust3r_ga_output_dir is not None:
                dust3r_ga_output_path = sample['dust3r_ga_output_path']
                with open(dust3r_ga_output_path, 'rb') as f:
                    dust3r_ga_output_and_gt_cameras = pickle.load(f)
                dust3r_ga_output = dust3r_ga_output_and_gt_cameras['dust3r_ga']

        """ load 2d annot for human optimization later; GT parameters """
        multiview_multiple_human_2d_cam_annot = {}
        for camera_name in selected_cameras:
            # Load image if it exists
            img_path = sample['annot_and_img_paths'][camera_name]['img_path']
            if os.path.exists(img_path):
                # load 2d pose annot
                pose2d_annot_path = sample['annot_and_img_paths'][camera_name]['pose2d_annot_path']
                pose2d_annot = np.load(pose2d_annot_path, allow_pickle=True)

                # load bbox annot
                bbox_annot_path = sample['annot_and_img_paths'][camera_name]['bbox_annot_path']
                if bbox_annot_path is not None:
                    bbox_annot = np.load(bbox_annot_path, allow_pickle=True)
                else:
                    bbox_annot = None

                # Store annotations for this camera
                # pose2d and bbox should always exist together, otherwise just not 2d annotation for that (camera, human) pair
                human_names_from_pose2d_annot = set([pose2d_annot[i]['human_name'] for i in range(len(pose2d_annot))])
                multiview_multiple_human_2d_cam_annot[camera_name] = defaultdict(dict)
                if bbox_annot is not None:
                    for j in range(len(bbox_annot)):
                        human_name = bbox_annot[j]['human_name']
                        if human_name not in human_names_from_pose2d_annot:
                            continue
                        bbox = bbox_annot[j]['bbox']
                        multiview_multiple_human_2d_cam_annot[camera_name][human_name] = {'bbox': bbox}
                
                for j in range(len(pose2d_annot)):
                    human_name = pose2d_annot[j]['human_name']
                    if human_name not in multiview_multiple_human_2d_cam_annot[camera_name]:
                        multiview_multiple_human_2d_cam_annot[camera_name][human_name] = {}
                    if pose2d_annot[j]['keypoints'].shape[0] != 133 or np.all(pose2d_annot[j]['keypoints'] == 0) or ('is_valid' in pose2d_annot[j] and not pose2d_annot[j]['is_valid']):
                        pose2d = None
                        del multiview_multiple_human_2d_cam_annot[camera_name][human_name] 
                        continue # just don't use this human
                    else:
                        pose2d = pose2d_annot[j]['keypoints']
                        multiview_multiple_human_2d_cam_annot[camera_name][human_name] = {'pose2d': pose2d}
                        if 'bbox' not in multiview_multiple_human_2d_cam_annot[camera_name][human_name]:
                            multiview_multiple_human_2d_cam_annot[camera_name][human_name]['bbox'] = pose2d_annot[j]['bbox']

                # Visualize the groundtruth 2D annotations per camera view per human
                # for human_name, annot in multiview_multiple_human_2d_cam_annot[camera_name].items():
                #     img = cv2.imread(img_path) # visualize the groundtruth keypoints and bboxes

                #     bbox = annot['bbox']
                #     pose2d = annot['pose2d']

                #     # Draw bounding box and keypoints on the image with human name
                #     if bbox is not None:
                #         x1, y1, x2, y2, conf = bbox
                #         img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                #         # Put human name text above bbox
                #         cv2.putText(img, human_name, (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                #     # Draw 2D keypoints
                #     if pose2d is not None:
                #         # Draw all keypoints
                #         for kp_idx in range(pose2d.shape[0]):
                #             x, y, conf = pose2d[kp_idx]
                #             img = cv2.circle(img, (int(x), int(y)), 2, (255, 0, 0), -1)
                #     cv2.imwrite(f'{camera_name}_{os.path.basename(img_path)[:-4]}_gt_keypoints_{human_name}.png', img)

        """ load 2D pose and 3D mesh predictions for human optimization later; Predicted parameters """
        multiview_multiple_human_2d_cam_pred = {}

        for camera_name in selected_cameras:
            # multiview_multiple_human_2d_cam_pred[camera_name] = {}
            # vit 2d pose pred path
            # '/scratch/one_month/current/lmueller/egohuman/camera_ready/01_tagging/001_tagging/processed_data/humanwithhand/cam01/__vitpose.pkl
            mono_multiple_human_2d_cam_pred_path = os.path.join(self.vitpose_hmr2_hamer_output_dir, self.big_seq_name_dict[seq.split('_')[1]], seq, 'processed_data/humanwithhand', f'{camera_name}', '__vitpose.pkl')
            with open(mono_multiple_human_2d_cam_pred_path, 'rb') as f:
                mono_multiple_human_2d_cam_pred_pose = pickle.load(f)
            # the saved frame keys are 1-indexed following the image names
            if f'{frame+1:05d}' not in mono_multiple_human_2d_cam_pred_pose.keys():
                print(f'Warning: no vitpose 2d pose pred for {camera_name} frame {frame+1:05d}. Skipping this frame.')
                continue
            mono_multiple_human_2d_cam_pred_pose = mono_multiple_human_2d_cam_pred_pose[f'{frame+1:05d}'] # (N, 133, 2+1)


            # load 2d bbox pred path
            mono_multiple_human_2d_cam_pred_bbox_path = os.path.join(self.vitpose_hmr2_hamer_output_dir, self.big_seq_name_dict[seq.split('_')[1]], seq, 'processed_data/humanwithhand', f'{camera_name}', '__bbox_with_score_used.pkl')
            with open(mono_multiple_human_2d_cam_pred_bbox_path, 'rb') as f:
                mono_multiple_human_2d_cam_pred_bbox = pickle.load(f)
            # the saved frame keys are 1-indexed following the image names
            if f'{frame+1:05d}' not in mono_multiple_human_2d_cam_pred_bbox.keys():
                print(f'Warning: no bbox pred for {camera_name} frame {frame+1:05d}. Skipping this frame.')
                continue
            mono_multiple_human_2d_cam_pred_bbox = mono_multiple_human_2d_cam_pred_bbox[f'{frame+1:05d}'] # (N, 4+1)

            # 3d mesh annot path
            #  '/scratch/one_month/current/lmueller/egohuman/camera_ready/01_tagging/001_tagging/processed_data/humanwithhand/cam01/__hmr2_hamer_smplx.pkl'
            mono_multiple_human_3d_cam_pred_path = os.path.join(self.vitpose_hmr2_hamer_output_dir, self.big_seq_name_dict[seq.split('_')[1]], seq, 'processed_data/humanwithhand', f'{camera_name}', '__hmr2_hamer_smplx.pkl')
            with open(mono_multiple_human_3d_cam_pred_path, 'rb') as f:
                mono_multiple_human_3d_cam_pred = pickle.load(f)
            # the saved frame keys are 1-indexed following the image names
            if f'{frame+1:05d}' not in mono_multiple_human_3d_cam_pred.keys():
                print(f'Warning: no hmr2_hamer 3d mesh pred for {camera_name} frame {frame+1:05d}. Skipping this frame.')
                continue
            mono_multiple_human_3d_cam_pred = mono_multiple_human_3d_cam_pred[f'{frame+1:05d}']['params'] # dictionary of human parameters 

            assert len(mono_multiple_human_2d_cam_pred_pose) == len(mono_multiple_human_2d_cam_pred_bbox) == len(mono_multiple_human_3d_cam_pred['body_pose']), "The number of 2D pose predictions, 2D bbox predictions, and 3D mesh predictions should match!"
            # sanitize the mono predictions by comparing the 2D keypoints. Do NMS on the 2D keypoints and return the unique indices for the list
            unique_mono_pred_indices = nms_unique_pose2d_indices(mono_multiple_human_2d_cam_pred_pose)
            mono_multiple_human_2d_cam_pred_pose = mono_multiple_human_2d_cam_pred_pose[unique_mono_pred_indices]
            mono_multiple_human_2d_cam_pred_bbox = mono_multiple_human_2d_cam_pred_bbox[unique_mono_pred_indices]
            mono_multiple_human_3d_cam_pred = {key: mono_multiple_human_3d_cam_pred[key][unique_mono_pred_indices] for key in mono_multiple_human_3d_cam_pred.keys()}

            # Visualization of pose2d and bbox predictions
            # pred_pose2d = mono_multiple_human_2d_cam_pred_pose
            # # Draw predicted keypoints in green
            # img_path = sample['annot_and_img_paths'][camera_name]['img_path']
            # img = cv2.imread(img_path)
            # for i in range(len(pred_pose2d)):
            #     pred = pred_pose2d[i]
            #     # Draw index and assigned name
            #     img = cv2.putText(img, f"{i}", 
            #                     (int(pred[0, 0]), int(pred[0, 1])), 
            #                     cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            #     # Draw keypoints
            #     for kp in pred:
            #         if kp[2] > 0:  # Only draw if confidence > 0
            #             img = cv2.circle(img, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), -1)

            #     # draw bbox
            #     x1, y1, x2, y2, conf = mono_multiple_human_2d_cam_pred_bbox[i]
            #     img = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # # save the image with the predicted keypoints
            # file_name = os.path.basename(img_path)[:-4]
            # cv2.imwrite(f'{camera_name}_{file_name}_pred_keypoints.png', img)

            # assign the human names to the 2d pose, 2d bbox, and 3d mesh predictions by associating the predicted 2D keypoints with the Egohuman dataset 2D keypoints
            multiple_human_2d_cam_annot_human_names = sorted(list(multiview_multiple_human_2d_cam_annot[camera_name].keys()))
            multiple_human_2d_cam_annot_pose2d = np.array([multiview_multiple_human_2d_cam_annot[camera_name][human_name]['pose2d'] for human_name in multiple_human_2d_cam_annot_human_names])
            multiple_human_2d_cam_annot_bbox = np.array([multiview_multiple_human_2d_cam_annot[camera_name][human_name]['bbox'] for human_name in multiple_human_2d_cam_annot_human_names])
           
            # Use bbox to assign human names to the predictions
            mono_pred_human_names = assign_human_names_to_mono_predictions(mono_multiple_human_2d_cam_pred_bbox, multiple_human_2d_cam_annot_bbox, 
                                                                           mono_multiple_human_2d_cam_pred_pose, multiple_human_2d_cam_annot_pose2d, \
                                                                           multiple_human_2d_cam_annot_human_names, \
                                                                           self.egohumans_image_size_tuple, \
                                                                        #    cam_name = camera_name, img_path = sample['annot_and_img_paths'][camera_name]['img_path'] \
                                                                           ) 
            mono_pred_output_dict = {mono_pred_human_names[i]: 
                                    {
                                        'original_index_in_mono_vitpose_pred': unique_mono_pred_indices[i],
                                        'pose2d': mono_multiple_human_2d_cam_pred_pose[i],
                                        'bbox': mono_multiple_human_2d_cam_pred_bbox[i],
                                        'params': {key: mono_multiple_human_3d_cam_pred[key][i] for key in mono_multiple_human_3d_cam_pred.keys()}
                                    } for i in range(len(mono_pred_human_names)) if mono_pred_human_names[i] is not None}
            multiview_multiple_human_2d_cam_pred[camera_name] = mono_pred_output_dict

            # SAVE_DIR
            mono_multiple_human_sanitized_save_dir = os.path.join(self.vitpose_hmr2_hamer_output_dir, self.big_seq_name_dict[seq.split('_')[1]], seq, 'processed_data/humanwithhand', f'{camera_name}', 'identified_predictions')
            if not os.path.exists(mono_multiple_human_sanitized_save_dir):
                os.makedirs(mono_multiple_human_sanitized_save_dir)
            mono_multiple_human_sanitized_save_path = os.path.join(mono_multiple_human_sanitized_save_dir, f'__hongsuk_identified_vitpose_bbox_smplx_frame{frame+1:05d}.pkl')
            with open(mono_multiple_human_sanitized_save_path, 'wb') as f:
                pickle.dump(mono_pred_output_dict, f)
            print(f'Saved sanitized predictions to {mono_multiple_human_sanitized_save_path}')

        """ get MultiHMR output; Predicted parameters """
        if self.dust3r_raw_output_dir is not None and self.dust3r_ga_output_dir is not None and self.multihmr_output_path is not None:
            multihmr_output = self.multihmr_output[f'{seq}_{frame}_{"".join(selected_cameras)}']['first_cam_humans']

            # assign the human names to the multihmr output
            # Later, you can replace the multihmr_2d_pred with ViTPose 2D keypoints output
            first_cam_human_names = list(multiview_multiple_human_2d_cam_annot[first_cam].keys())
            multihmr_2d_pred = [human['j2d'].cpu().numpy() for human in multihmr_output]
            egohumans_2d_annot = [multiview_multiple_human_2d_cam_annot[first_cam][human_name]['pose2d'] for human_name in first_cam_human_names]
            multihmr_output_human_names = assign_human_names_to_multihmr_output(multihmr_first_cam_affine_matrix, multihmr_2d_pred, egohumans_2d_annot, first_cam_human_names, \
                                                                                '')#sample['annot_and_img_paths'][first_cam]['img_path']) 
            multihmr_output_dict = {multihmr_output_human_names[i]: multihmr_output[i] for i in range(len(multihmr_output))}

        # Load all required data
        # Data dictionary contains:
        # - multiview_images: Dict[camera_name -> Dict] containing preprocessed images for DUSt3R
        #     - img: preprocessed image torch tensor; shape (3, 288, 512)
        #     - true_shape: original image dimensions, which is (288,512); shape (2, )
        # - multiview_affine_transforms: Dict[camera_name -> np.ndarray] containing affine matrices, np array shape (2, 3)
        #     mapping from cropped to original image coordinates
        # - multiview_multiple_human_2d_cam_annot: Dict[camera_name -> Dict[human_name -> Dict]] containing 2D annotations
        #     - pose2d: 2D keypoint coordinates, np array shape (133, 2+1), x, y, confidence
        #     - bbox: Bounding box coordinates, np array shape (4+1, ), x1, y1, x2, y2, confidence

        # - world_multiple_human_3d_annot: Dict[human_name -> Dict] containing 3D world smpl parameters, this is for evaluation, you can put GT 3D joints here instead
        # - camera_parameters: Dict[camera_name -> Dict] containing camera parameters

        # - sequence: Sequence name/ID
        # - frame: Frame number
        data = {
            'multiview_images': multiview_images, # for Dust3R network inference
            'multiview_affine_transforms': multiview_affine_transforms,
            'multiview_multiple_human_2d_cam_annot': multiview_multiple_human_2d_cam_annot, # groundtruth 2D annotations
            'multiview_cameras': cameras, # groundtruth camera parameters
            'world_multiple_human_3d_annot': world_multiple_human_3d_annot, # groundtruth 3D annotations
            'multihmr_first_cam_input_image': multihmr_first_cam_input_image, # for MultiHMR inference
            'multihmr_intrinsic': multihmr_first_cam_intrinsic, # for MultiHMR inference
            'sequence': seq,
            'frame': frame
        }

        # add dust3r network output if it exists
        if self.dust3r_raw_output_dir is not None:
            data['dust3r_network_output'] = dust3r_network_output
        if self.dust3r_raw_output_dir is not None and self.dust3r_ga_output_dir is not None:
            data['dust3r_ga_output'] = dust3r_ga_output
        if self.dust3r_raw_output_dir is not None and self.dust3r_ga_output_dir is not None and self.multihmr_output_path is not None:
            data['multihmr_output'] = multihmr_output_dict

        return data


def vec_image_from_cam(intrinsics, point_3d, eps=1e-9):
    fx, fy, cx, cy, k1, k2, k3, k4 = intrinsics

    x = point_3d[:, 0]
    y = point_3d[:, 1]
    z = point_3d[:, 2]

    a = x/z; b = y/z
    r = np.sqrt(a*a + b*b)
    theta = np.arctan(r)

    theta_d = theta * (1 + k1*theta**2 + k2*theta**4 + k3*theta**6 + k4*theta**8)
    x_prime = (theta_d/r)*a
    y_prime = (theta_d/r)*b

    u = fx*(x_prime + 0) + cx
    v = fy*y_prime + cy

    point_2d = np.concatenate([u.reshape(-1, 1), v.reshape(-1, 1)], axis=1)

    return point_2d

def nms_unique_pose2d_indices(pose2d, threshold=30):
    # threshold: pixel distance threshold
    # pose2d is a list of np.ndarray shape (133, 2+1)
    # do NMS on the 2D keypoints and return the unique indices for the list
    # compare the pose pairs by the euclidean distance between the 2D keypoints and remove the duplicate predictions
    # return the unique indices
    if len(pose2d) <= 1:
        return np.arange(len(pose2d))
    
    # Convert list of arrays to single array for easier indexing
    pose2d = np.array(pose2d)  # (N, 133, 3)
    
    # Get valid keypoints mask based on confidence
    valid_mask = pose2d[..., 2] > 0  # (N, 133)
    
    # Calculate pairwise distances between all poses
    N = len(pose2d)
    distances = np.zeros((N, N))
    
    for i in range(N):
        for j in range(i+1, N):
            # Get common valid keypoints between pose pairs
            common_valid = valid_mask[i] & valid_mask[j]  # (133,)
            
            if not np.any(common_valid):
                # If no common valid keypoints, set large distance
                distances[i,j] = distances[j,i] = float('inf')
                continue
                
            # Calculate mean euclidean distance between valid keypoints
            pose_i = pose2d[i, common_valid, :2]  # (K, 2) 
            pose_j = pose2d[j, common_valid, :2]  # (K, 2)
            dist = np.mean(np.sqrt(np.sum((pose_i - pose_j)**2, axis=1)))
            distances[i,j] = distances[j,i] = dist
    
    # Do NMS - keep poses that are far enough from each other
    keep_indices = []
    remaining = list(range(N))
    
    while remaining:
        # Get pose with highest average confidence
        confs = [np.mean(pose2d[i, valid_mask[i], 2]) for i in remaining]
        idx = remaining[np.argmax(confs)]
        keep_indices.append(idx)
        
        # Remove poses that are too close
        remaining = [j for j in remaining if distances[idx,j] > threshold]
    
    return np.array(keep_indices)


# assign human names to the mono predictions by associating the predicted bboxes with the Egohuman dataset bboxes
def compute_iou(bbox1, bbox2):
    # bbox1, bbox2: np.ndarray shape (4+1, ), 
    # each bbox: x1, y1, x2, y2, conf
    # return the IoU between the two bboxes
    
    # Get coordinates
    x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
    x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
    
    # Calculate intersection coordinates
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2) 
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    # Calculate areas
    area_1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area_2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate intersection area
    if x2_i < x1_i or y2_i < y1_i:
        # No intersection
        return 0.0
        
    area_i = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area_u = area_1 + area_2 - area_i
    
    # Return IoU
    return area_i / area_u

def assign_human_names_to_mono_predictions(pred_bboxes, gt_bboxes, pred_poses2d, gt_poses2d, gt_human_names, img_size, cam_name='', img_path=''):
    """
    Assign human names to predicted bboxes by matching them with ground truth bboxes
    
    Args:
        pred_bboxes: np.ndarray shape (M, 4+1) - predicted bboxes
        gt_bboxes: np.ndarray shape (N, 4+1) - ground truth bboxes
        pred_poses2d: np.ndarray shape (M, 133, 2+1) - predicted 2D keypoints
        gt_poses2d: np.ndarray shape (N, 133, 2+1) - ground truth 2D keypoints
        gt_human_names: length N list of human names matching gt_bboxes order
        img_size: tuple of (height, width)
        img_path: optional path to save visualization
    
    Returns:
        pred_bboxes_human_names: list of human names matching pred_bboxes order
    """
    assert len(gt_bboxes) == len(gt_human_names), "Number of GT bboxes must match number of human names"
    
    # Convert predictions to list if numpy array
    if isinstance(pred_bboxes, np.ndarray):
        pred_bboxes = [pred_bboxes[i] for i in range(len(pred_bboxes))]

    # Build cost matrix for Hungarian algorithm
    cost_matrix = np.zeros((len(pred_bboxes), len(gt_bboxes)))
    for i in range(len(pred_bboxes)):
        for j in range(len(gt_bboxes)):
            # confidence weights are the product of the confidence scores of the predicted and ground truth bboxes
            # confidence_weights = pred[4] * gt[4] # these are corrupted...
            # compute the IoU between the predicted and ground truth bboxes
            iou = compute_iou(pred_bboxes[i], gt_bboxes[j])
            
            # compute the euclidean distance between the predicted and ground truth 2D keypoints
            dist = np.sum(pred_poses2d[i, :, 2] * gt_poses2d[j, :, 2] * np.sqrt(np.sum((pred_poses2d[i, :, :2] - gt_poses2d[j, :, :2])**2, axis=1)))
            
            # normalize the distance by the size of the image
            img_area = img_size[0] * img_size[1]
            dist = dist / np.sqrt(img_area)

            if iou > 0.05:
                cost_matrix[i, j] = dist
            else:
                cost_matrix[i, j] = float(1e+10) # float('inf')

    # Run Hungarian algorithm to find optimal assignment
    # cost_matrix: np.ndarray shape (M, N)
    # maximize the cost_matrix to find the optimal assignment
    # row_indices, col_indices = linear_sum_assignment(cost_matrix, maximize=True)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # Assign human names based on matching
    pred_human_names = [None] * len(pred_bboxes)
    for idx in range(len(row_indices)): # len(row_indices) == len(col_indices
        r = row_indices[idx]
        c = col_indices[idx]
        pred_human_names[r] = gt_human_names[c]

    # for i in range(len(pred_human_names)):
    #     if pred_human_names[i] is not None:
    #         print(f'For camera {cam_name}, predicted index: {i}, assigned egohumans human name: {pred_human_names[i]}')

    # Optionally visualize the matches
    if img_path != '':
        output_dir = './inspect_egohumans_reid'
        os.makedirs(output_dir, exist_ok=True)
        # Draw predicted keypoints in green
        img = cv2.imread(img_path)
        for i in range(len(pred_human_names)):
            if pred_human_names[i] is not None:
                pred = pred_bboxes[i]
                # Draw index and assigned name
                img = cv2.putText(img, f"{i}:{pred_human_names[i]}", 
                            (int(pred[0] - 10), int(pred[1] - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
                # Draw bbox
                img = cv2.rectangle(img, (int(pred[0]), int(pred[1])), (int(pred[2]), int(pred[3])), (0, 255, 0), 2)

        # save the image with the predicted bboxes
        file_name = os.path.basename(img_path)[:-4]
        seq_name = img_path.split('/')[-5]
        cv2.imwrite(os.path.join(output_dir, f'{seq_name}_{cam_name}_{file_name}_pred_matching.png'), img)

        # Draw ground truth keypoints in red
        img = cv2.imread(img_path)   
        for i, gt in enumerate(gt_bboxes):
            # Draw index and name
            img = cv2.putText(img, f"{i}:{gt_human_names[i]}", 
                            (int(gt[0] - 10), int(gt[1] - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            # Draw bbox
            img = cv2.rectangle(img, (int(gt[0]), int(gt[1])), (int(gt[2]), int(gt[3])), (0, 0, 255), 2)
        
        file_name = os.path.basename(img_path)[:-4]
        seq_name = img_path.split('/')[-5]
        cv2.imwrite(os.path.join(output_dir, f'{seq_name}_{cam_name}_{file_name}_gt_matching.png'), img)

    return pred_human_names

# assign human names to the multihmr output by associating the predicted 2D keypoints with the Egohuman dataset 2D keypoints
def assign_human_names_to_multihmr_output(multihmr_affine_matrix, multihmr_2d_pred_list, egohumans_2d_annot_list, egohumans_human_names, img_path=''):
    assert len(multihmr_2d_pred_list) == len(egohumans_2d_annot_list) == len(egohumans_human_names), "The number of predictions, annotations, and human names should match!"
    # multihmr_affine_matrix: np.ndarray shape (2, 3)
    # multihmr_2d_pred_list: list of np.ndarray shape (127, 2)
    # egohumans_2d_annot_list: list of np.ndarray shape (133, 2)
    # egohumans_human_names: list of human names, the indices should match the order of egohumans_2d_annot_list

    # apply the affine transformation to the multihmr 2d predictions
    multihmr_2d_pred_transformed_list = []
    for i in range(len(multihmr_2d_pred_list)):
        homogeneous_coords = np.hstack([multihmr_2d_pred_list[i], np.ones((multihmr_2d_pred_list[i].shape[0], 1))])
        multihmr_2d_pred_transformed = (multihmr_affine_matrix @ homogeneous_coords.T)[:2, :].T

        # map the multihmr 2d pred to the COCO_WHOLEBODY_KEYPOINTS
        placeholder_multihmr_2d_pred_transformed = np.zeros((len(COCO_WHOLEBODY_KEYPOINTS), 2))
        for i, joint_name in enumerate(COCO_WHOLEBODY_KEYPOINTS):
            if joint_name in SMPLX_JOINT_NAMES:
                # print(f'{joint_name}')
                placeholder_multihmr_2d_pred_transformed[i, :2] = multihmr_2d_pred_transformed[SMPLX_JOINT_NAMES.index(joint_name)]
                # placeholder_multihmr_2d_pred_transformed[i, 2] = 1

        multihmr_2d_pred_transformed_list.append(placeholder_multihmr_2d_pred_transformed)

    def draw_2d_keypoints(img, keypoints, keypoints_name=None, color=(0, 255, 0), radius=3):
        for i, keypoint in enumerate(keypoints):
            if keypoints_name is not None:
                if keypoints_name[i] in ['nose1', 'nose2', 'nose3', 'nose4']:
                    img = cv2.putText(img, keypoints_name[i], (int(keypoint[0]), int(keypoint[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            img = cv2.circle(img, (int(keypoint[0]), int(keypoint[1])), radius, color, -1)
        return img
    
    if img_path != '':
        img = cv2.imread(img_path)
        # draw the predicted and annotated 2D keypoints on the image
        for i in range(len(multihmr_2d_pred_transformed_list)):
            # Draw index
            img = cv2.putText(img, str(i), (int(multihmr_2d_pred_transformed_list[i][0, 0]), int(multihmr_2d_pred_transformed_list[i][0, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            img = draw_2d_keypoints(img, multihmr_2d_pred_transformed_list[i], keypoints_name=COCO_WHOLEBODY_KEYPOINTS, color=(0, 255, 0), radius=3)
        for i in range(len(egohumans_2d_annot_list)):
            # Draw index
            img = cv2.putText(img, str(i), (int(egohumans_2d_annot_list[i][0, 0]), int(egohumans_2d_annot_list[i][0, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            # Draw the name too
            img = cv2.putText(img, egohumans_human_names[i], (int(egohumans_2d_annot_list[i][0, 0]), int(egohumans_2d_annot_list[i][0, 1])), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
            img = draw_2d_keypoints(img, egohumans_2d_annot_list[i], color=(0, 0, 255), radius=3)
        cv2.imwrite('multihmr_2d_pred_transformed.png', img)

    # build a cost matrix for association by hungarian algorithm
    cost_matrix = np.zeros((len(multihmr_2d_pred_transformed_list), len(egohumans_2d_annot_list)))
    for i, pred in enumerate(multihmr_2d_pred_transformed_list):
        for j, annot in enumerate(egohumans_2d_annot_list):
            # compute the euclidean distance between the predicted and annotated 2D keypoints   
            cost_matrix[i, j] = np.sum(annot[:, 2:] * np.sqrt(np.sum((pred - annot[:, :2])**2, axis=1)))    
    
    # run the hungarian algorithm to find the optimal assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    # assign the human names to the multihmr output
    # THIS IS WRONG - Hongsuk!@!!!!!!!!f Why don't you use the row indices?
    multihmr_output_human_names = [egohumans_human_names[j] for j in col_indices]
    # print the index of multihmr output and its assigned egohumans human name
    # for i in range(len(multihmr_output_human_names)):
    #     print(f'multihmr output index: {i}, assigned egohumans human name: {multihmr_output_human_names[i]}')

    return multihmr_output_human_names

# get the affine transform matrix from cropped image to original image
# this is just to get the affine matrix (crop image to original image) - Hongsuk
def get_dust3r_affine_transform(file, size=512, square_ok=False):
    img = PIL.Image.open(file)
    original_width, original_height = img.size
    
    # Step 1: Resize
    S = max(img.size)
    if S > size:
        interp = PIL.Image.LANCZOS
    else:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*size/S)) for x in img.size)
    
    # Calculate center of the resized image
    cx, cy = size // 2, size // 2

    # Step 2: Crop
    halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
    if not square_ok and new_size[0] == new_size[1]:
        halfh = 3*halfw//4
        
    # Calculate the total transformation
    scale_x = new_size[0] / original_width
    scale_y = new_size[1] / original_height
    
    translate_x = (cx - halfw) / scale_x
    translate_y = (cy - halfh) / scale_y
    
    affine_matrix = np.array([
        [1/scale_x, 0, translate_x],
        [0, 1/scale_y, translate_y]
    ])
    
    return affine_matrix

# open image for MultiHMR inference
def get_multihmr_input_image(img_path, img_size):
    """ Open image at path, resize and pad """

    # Open and reshape
    img_pil = Image.open(img_path).convert('RGB')
    
    # Get original size
    original_width, original_height = img_pil.size

    # reisze to the target size while keeping the aspect ratio
    img_pil = ImageOps.contain(img_pil, (img_size,img_size)) 

    # Get new size
    new_width, new_height = img_pil.size
    # Calculate scaling factors
    scale_x = original_width / new_width
    scale_y = original_height / new_height

    # Keep a copy for visualisations.
    img_pil_bis = ImageOps.pad(img_pil.copy(), size=(img_size,img_size), color=(255, 255, 255)) # image is keep centered
    img_pil = ImageOps.pad(img_pil, size=(img_size,img_size)) # pad with zero on the smallest side
    
    # Get new size
    padded_new_width, padded_new_height = img_pil_bis.size
    pad_width = (new_width - padded_new_width) / 2
    pad_height = (new_height - padded_new_height) / 2
    
    # Calculate translation
    translate_x = pad_width * scale_x
    translate_y = pad_height * scale_y
    
    # Create the affine transformation matrix
    affine_matrix = np.array([
        [scale_x, 0, translate_x],
        [0, scale_y, translate_y]
    ])

    # Go to numpy 
    resize_img = np.asarray(img_pil)

    # Normalize and go to torch.
    resize_img = normalize_rgb(resize_img)
    return resize_img, img_pil_bis, affine_matrix

def get_multihmr_camera_parameters(img_size, fov=60, p_x=None, p_y=None):
    """ Given image size, fov and principal point coordinates, return K the camera parameter matrix"""
    K = torch.eye(3)
    # Get focal length.
    focal = get_focalLength_from_fieldOfView(fov=fov, img_size=img_size)
    K[0,0], K[1,1] = focal, focal

    # Set principal point
    if p_x is not None and p_y is not None:
            K[0,-1], K[1,-1] = p_x * img_size, p_y * img_size
    else:
            K[0,-1], K[1,-1] = img_size//2, img_size//2

    return K

def create_dataloader(data_root, dust3r_raw_output_dir=None, dust3r_ga_output_dir=None, multihmr_output_path=None, batch_size=8, split='train', num_workers=4, subsample_rate=10, cam_names=None, num_of_cams=None, selected_big_seq_list=[]):
    """
    Create a dataloader for the multiview human dataset
    
    Args:
        data_root (str): Root directory of the dataset
        batch_size (int): Batch size for the dataloader
        split (str): 'train', 'val', or 'test'
        num_workers (int): Number of workers for data loading
        
    Returns:
        DataLoader: PyTorch dataloader for the dataset
    """
    dataset = EgoHumansDataset(
        data_root=data_root,
        dust3r_raw_output_dir=dust3r_raw_output_dir,
        dust3r_ga_output_dir=dust3r_ga_output_dir,
        multihmr_output_path=multihmr_output_path,
        split=split,
        subsample_rate=subsample_rate,
        cam_names=cam_names,
        num_of_cams=num_of_cams,
        selected_big_seq_list=selected_big_seq_list
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return dataset, dataloader


if __name__ == '__main__':
    from tqdm import tqdm
    # data_root = '/home/hongsuk/projects/egohumans/data'
    # dust3r_output_path =  '/home/hongsuk/projects/dust3r/outputs/egohumans/dust3r_network_output_30:11:10.pkl'
    # dust3r_ga_output_path = '/home/hongsuk/projects/dust3r/outputs/egohumans/dust3r_ga_output_02:20:29.pkl'
    # multihmr_output_path = '/home/hongsuk/projects/dust3r/outputs/egohumans/multihmr_output_30:23:17.pkl'
    # dataset, dataloader = create_dataloader(data_root, dust3r_output_path=dust3r_output_path, dust3r_ga_output_path=dust3r_ga_output_path, multihmr_output_path=multihmr_output_path, batch_size=1, split='test', num_workers=0)
    
    # ['06_badminton']  #['07_tennis'] #  # #['01_tagging', '02_lego, 05_volleyball', '04_basketball', '03_fencing']
    selected_big_seq_list = ['06_badminton']# ['5_volleyball', '04_basketball'] # ['01_tagging', '02_lego', '03_fencing'] -> might stop because of scipy infinity bug  #['07_tennis'] -> might stop because of scipy infinity bug
    num_of_cams = None
    data_root = '/home/hongsuk/projects/dust3r/data/egohumans_data'
    dust3r_output_dir = None # f'/home/hongsuk/projects/dust3r/outputs/egohumans/dust3r_raw_outputs/num_of_cams{num_of_cams}'
    dust3r_ga_output_dir = None # f'/home/hongsuk/projects/dust3r/outputs/egohumans/dust3r_ga_outputs_and_gt_cameras/num_of_cams{num_of_cams}'
    dataset, dataloader = create_dataloader(data_root, dust3r_raw_output_dir=dust3r_output_dir, dust3r_ga_output_dir=dust3r_ga_output_dir, num_of_cams=num_of_cams, batch_size=1, split='test', num_workers=0, selected_big_seq_list=selected_big_seq_list)

    dataset_len = len(dataset)
    step = 1
    for i in tqdm(range(0, dataset_len, step)):
        item = dataset.get_single_item(i)

