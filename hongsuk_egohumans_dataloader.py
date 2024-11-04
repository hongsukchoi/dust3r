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

from dust3r.utils.image import load_images as dust3r_load_images
from multihmr.utils import normalize_rgb, get_focalLength_from_fieldOfView
from multihmr.utils import get_smplx_joint_names

from hongsuk_joint_names import COCO_WHOLEBODY_KEYPOINTS, SMPLX_JOINT_NAMES

class EgoHumansDataset(Dataset):
    def __init__(self, data_root, dust3r_output_path=None, dust3r_ga_output_path=None, multihmr_output_path=None, split='train', subsample_rate=10, cam_names=None):
        """
        Args:
            data_root (str): Root directory of the dataset
            dust3r_output_path (str): Path to the dust3r output
            dust3r_ga_output_path (str): Path to the dust3r global alignment output
            multihmr_output_path (str): Path to the multihmr output
            split (str): 'train', 'val', or 'test'
        """
        self.data_root = data_root 
        self.split = split
        self.dust3r_image_size = 512
        self.multihmr_image_size = 896
        self.subsample_rate = subsample_rate

        # choose camera names
        if cam_names is None:
            self.camera_names = ['cam01', 'cam02', 'cam03', 'cam04']
        else:
            self.camera_names = cam_names
        # choose only a few sequence for testing else all sequences
        self.selected_small_seq_name_list = []  # ex) ['001_tagging', '002_tagging']

        # Load dust3r network output
        self.dust3r_output_path = dust3r_output_path
        if self.dust3r_output_path is not None:
            self.dust3r_output = pickle.load(open(self.dust3r_output_path, 'rb'))
            for output_name, output in self.dust3r_output.items():
                # output_name is like '001_tagging_0_cam01cam02cam03cam04'
                small_seq_name = '_'.join(output_name.split('_')[0:2])
                camera_names = output_name.split('_')[-1]
                assert ''.join(self.camera_names) == camera_names, f'Camera names in self.camera_names and dust3r output do not match: {self.camera_names} vs {camera_names}'
                self.selected_small_seq_name_list.append(small_seq_name)

        """ > Structure + Humans + Cameras optimization - Hongsuk (optional): """
        # Load dust3r global alignment output
        self.dust3r_ga_output_path = dust3r_ga_output_path
        if self.dust3r_output_path is not None and self.dust3r_ga_output_path is not None:
            dust3r_ga_output = pickle.load(open(self.dust3r_ga_output_path, 'rb'))
            self.dust3r_ga_output = {} # {output_name: output}
            no_ga_output_names = []
            for output_name in self.dust3r_output.keys():
                if output_name not in dust3r_ga_output.keys():
                    no_ga_output_names.append(output_name)
                else:
                    self.dust3r_ga_output[output_name] = dust3r_ga_output[output_name]
            if len(no_ga_output_names) > 0:
                print(f'Warning: {no_ga_output_names} does not have global alignment output')
        # Load multihmr output
        self.multihmr_output_path = multihmr_output_path
        if self.dust3r_output_path is not None and self.dust3r_ga_output_path is not None and self.multihmr_output_path is not None:
            multihmr_output = pickle.load(open(self.multihmr_output_path, 'rb'))
            self.multihmr_output = {} # {output_name: output}
            no_multihmr_output_names = []
            for output_name in self.dust3r_output.keys():
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
        self.big_seq_list = sorted(glob.glob(os.path.join(self.data_root, '*')))
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
                    print(f'Error loading annot for {small_seq}') # there is no smpl annot for this sequence

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

                
                # check whether the cameras.keys() are superset of the self.camera_names
                # if not, skip this frame
                selected_cameras = sorted([cam for cam in annot['cameras'].keys() if cam in self.camera_names ])
                if len(selected_cameras) != len(self.camera_names):
                    # print(f'Warning: {small_seq} does not have all the cameras in self.camera_names; skipping this frame')
                    continue

                # check whether the dust3r is given and the dust3r output exists; this is fore global alignment step of Dust3r
                if self.dust3r_output_path is not None and f'{small_seq}_{frame}_{"".join(selected_cameras)}' not in self.dust3r_output.keys():
                    continue

                # add camera data
                # per_frame_data['cameras'] = {cam: annot['cameras'][cam] for cam in selected_cameras} # dictionrary of camera names and their parameters
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
        selected_cameras = sorted([cam for cam in cameras.keys() if cam in self.camera_names])

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
        multihmr_first_cam_input_image, _, multihmr_first_cam_affine_matrix = get_multihmr_input_image(sample['annot_and_img_paths'][first_cam]['img_path'], self.multihmr_image_size)
        multihmr_first_cam_intrinsic = get_multihmr_camera_parameters(self.multihmr_image_size)

        """ get dust3r output; Predicted parameters """
        if self.dust3r_output_path is not None:
            dust3r_network_output = self.dust3r_output[f'{seq}_{frame}_{"".join(selected_cameras)}']['output']
            del dust3r_network_output['loss']
        if self.dust3r_output_path is not None and self.dust3r_ga_output_path is not None:
            dust3r_ga_output = self.dust3r_ga_output[f'{seq}_{frame}_{"".join(selected_cameras)}']['dust3r_ga']

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
                multiview_multiple_human_2d_cam_annot[camera_name] = {}
                for human_idx in range(len(pose2d_annot)):
                    if 'is_valid' in pose2d_annot[human_idx] and not pose2d_annot[human_idx]['is_valid']:
                        continue

                    human_name = pose2d_annot[human_idx]['human_name']
                    pose2d = pose2d_annot[human_idx]['keypoints']
                    if bbox_annot is not None:
                        bbox = bbox_annot[human_idx]['bbox']
                    else:
                        bbox = pose2d_annot[human_idx]['bbox']
                    
                    multiview_multiple_human_2d_cam_annot[camera_name][human_name] = {
                        'pose2d': pose2d,
                        'bbox': bbox
                    }

        """ get MultiHMR output; Predicted parameters """
        if self.dust3r_output_path is not None and self.dust3r_ga_output_path is not None and self.multihmr_output_path is not None:
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
        #     - bbox: Bounding box coordinates, np array shape (4+1, ), x, y, w, h, confidence

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
        if self.dust3r_output_path is not None:
            data['dust3r_network_output'] = dust3r_network_output
        if self.dust3r_output_path is not None and self.dust3r_ga_output_path is not None:
            data['dust3r_ga_output'] = dust3r_ga_output
        if self.dust3r_output_path is not None and self.dust3r_ga_output_path is not None and self.multihmr_output_path is not None:
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

def create_dataloader(data_root, dust3r_output_path=None, dust3r_ga_output_path=None, multihmr_output_path=None, batch_size=8, split='train', num_workers=4, subsample_rate=10, cam_names=None):
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
        dust3r_output_path=dust3r_output_path,
        dust3r_ga_output_path=dust3r_ga_output_path,
        multihmr_output_path=multihmr_output_path,
        split=split,
        subsample_rate=subsample_rate,
        cam_names=cam_names
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
    data_root = '/home/hongsuk/projects/egohumans/data'
    dust3r_output_path =  '/home/hongsuk/projects/dust3r/outputs/egohumans/dust3r_network_output_30:11:10.pkl'
    dust3r_ga_output_path = '/home/hongsuk/projects/dust3r/outputs/egohumans/dust3r_ga_output_02:20:29.pkl'
    multihmr_output_path = '/home/hongsuk/projects/dust3r/outputs/egohumans/multihmr_output_30:23:17.pkl'
    dataset, dataloader = create_dataloader(data_root, dust3r_output_path=dust3r_output_path, dust3r_ga_output_path=dust3r_ga_output_path, multihmr_output_path=multihmr_output_path, batch_size=1, split='test', num_workers=0)
    item = dataset.get_single_item(0)
    # for data in dataloader:
    #     import pdb; pdb.set_trace()
    #     break