# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import os 
import tyro
import pickle
from pathlib import Path

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ['EGL_DEVICE_ID'] = '0'

import sys
from argparse import ArgumentParser
import random
import pickle as pkl
import numpy as np
from PIL import Image, ImageOps
import torch
from tqdm import tqdm
import time
import cv2

from multihmr.utils import normalize_rgb, render_meshes, get_focalLength_from_fieldOfView, demo_color as color, print_distance_on_image, render_side_views, create_scene, MEAN_PARAMS, CACHE_DIR_MULTIHMR, SMPLX_DIR
from multihmr.model import Model
from pathlib import Path
import warnings



def open_image(img_path, img_size, device=torch.device('cuda')):
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
    x = torch.from_numpy(resize_img).unsqueeze(0).to(device)
    return x, img_pil_bis, affine_matrix

def get_camera_parameters(img_size, fov=60, p_x=None, p_y=None, device=torch.device('cuda')):
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

    # Add batch dimension
    K = K.unsqueeze(0).to(device)
    return K

def load_model(ckpt_path, device=torch.device('cuda')):

    # Load weights
    print("Loading model")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Get arguments saved in the checkpoint to rebuild the model
    kwargs = {}
    for k,v in vars(ckpt['args']).items():
            kwargs[k] = v

    # Build the model.
    kwargs['type'] = ckpt['args'].train_return_type
    kwargs['img_size'] = ckpt['args'].img_size[0]
    model = Model(**kwargs).to(device)

    # Load weights into model.
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    print("Weights have been loaded")

    return model

def forward_model(model, input_image, camera_parameters,
                  det_thresh=0.3,
                  nms_kernel_size=1,
                 ):
        
    """ Make a forward pass on an input image and camera parameters. """
    
    # Forward the model.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            humans = model(input_image, 
                           is_training=False, 
                           nms_kernel_size=int(nms_kernel_size),
                           det_thresh=det_thresh,
                           K=camera_parameters)

    return humans

def overlay_human_meshes(humans, K, model, img_pil, unique_color=False):

    # Color of humans seen in the image.
    _color = [color[0] for _ in range(len(humans))] if unique_color else color
    
    # Get focal and princpt for rendering.
    focal = np.asarray([K[0,0,0].cpu().numpy(),K[0,1,1].cpu().numpy()])
    princpt = np.asarray([K[0,0,-1].cpu().numpy(),K[0,1,-1].cpu().numpy()])

    # Get the vertices produced by the model.
    verts_list = [humans[j]['v3d'].cpu().numpy() for j in range(len(humans))]
    faces_list = [model.smpl_layer['neutral_10'].bm_x.faces for j in range(len(humans))]

    # Render the meshes onto the image.
    pred_rend_array = render_meshes(np.asarray(img_pil), 
                                    verts_list,
                                    faces_list,
                                    {'focal': focal, 'princpt': princpt},
                                    alpha=1.0,
                                    color=_color)

    return pred_rend_array, _color

def visualize_2d_keypoints_in_org_img(org_img: np.ndarray, joints_2d: np.ndarray, affine_matrix: np.ndarray, bbox: list, human_idx: int, score: float):
    """ Transform the 2d keypoints to the original image size and visualize them """
    def transform_keypoints(keypoints, affine_matrix):
        # Ensure keypoints is a numpy array
        keypoints = np.array(keypoints)
        
        # Add a column of ones to make homogeneous coordinates
        homogeneous_keypoints = np.column_stack((keypoints, np.ones(len(keypoints))))
        
        # Apply the transformation
        transformed_keypoints = np.dot(affine_matrix, homogeneous_keypoints.T).T
        
        # Round to nearest integer for pixel coordinates
        transformed_keypoints = np.round(transformed_keypoints).astype(int)
        
        return transformed_keypoints

    joints_2d = transform_keypoints(joints_2d, affine_matrix)
    # convert xywh bbox to (N,2) format
    bbox = np.array([[bbox[0], bbox[1]], [bbox[0] + bbox[2], bbox[1] + bbox[3]]])
    bbox = transform_keypoints(bbox, affine_matrix)
    # convert back to xywh format
    bbox = np.array([bbox[0][0], bbox[0][1], bbox[1][0] - bbox[0][0], bbox[1][1] - bbox[0][1]])

    # Visualize the keypoints on the original image
    org_img = org_img.copy()
    for joint in joints_2d:
        org_img = cv2.circle(org_img, (int(joint[0]), int(joint[1])), 3, (0, 255, 0), -1) 

    org_img = cv2.rectangle(org_img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255), 2)
    org_img = cv2.putText(org_img, f'{human_idx}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    org_img = cv2.putText(org_img, f'{score:.2f}', (int(bbox[0]), int(bbox[1]) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return org_img

def main(img_folder: str, model_path: str = '/home/hongsuk/projects/SimpleCode/multiview_world/models/multiHMR/multiHMR_896_L.pt', output_dir: str = './outputs', fov: float = 60, det_thresh: float = 0.3, nms_kernel_size: int = 3, unique_color: int = 0, vis_keypoints: bool = False, render: bool = False):
    if img_folder.endswith('/'):
        img_folder_name = os.path.basename(img_folder[:-1])
    else:
        img_folder_name = os.path.basename(img_folder) 
    output_dir = os.path.join(output_dir, img_folder_name)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Input images
    suffixes = ('.jpg', '.jpeg', '.png', '.webp')
    l_img_path = [file for file in os.listdir(img_folder) if file.endswith(suffixes) and file[0] != '.']

    # Loading
    model = load_model(model_path)

    # All images
    humans_dict = {}
    affine_matrix_dict = {}
    for i, img_path in enumerate(tqdm(l_img_path)):
        # Get input in the right format for the model
        img_size = model.img_size
        x, img_pil_nopad, affine_matrix = open_image(os.path.join(img_folder, img_path), img_size)
        # Get camera parameters
        p_x, p_y = None, None
        K = get_camera_parameters(model.img_size, fov=fov, p_x=p_x, p_y=p_y)

        # Make model predictions
        humans = forward_model(model, x, K, det_thresh=det_thresh, nms_kernel_size=nms_kernel_size)
        # humans[0].keys: ['scores', 'loc', 'transl', 'transl_pelvis', 'rotvec', 'expression', 'shape', 'v3d', 'j3d', 'j2d']
        img_name = os.path.basename(img_path).split('.')[0]
        humans_dict[img_name] = humans
        affine_matrix_dict[img_name] = affine_matrix

        for human_idx, human in enumerate(humans):
            # Get bounding box from 2D joints
            j2d = human['j2d'].cpu().numpy()
            x_min, y_min = j2d.min(axis=0)
            x_max, y_max = j2d.max(axis=0)
            
            # Calculate width and height
            width = x_max - x_min
            height = y_max - y_min
            
            # Create bounding box (x, y, w, h)
            bbox = [x_min, y_min, width, height]
            
            # Save bounding box to the human dictionary
            human['bbox'] = bbox

        if render:
            # Superimpose predicted human meshes to the input image.
            img_array = np.asarray(img_pil_nopad)
            img_pil_visu= Image.fromarray(img_array)
            pred_rend_array, _color = overlay_human_meshes(humans, K, model, img_pil_visu, unique_color=unique_color)
            # Save the rendered image
            output_filename = f'debug_multihmr_rendered_{img_name}.png'
            output_path = os.path.join(output_dir, output_filename)
            Image.fromarray(pred_rend_array).save(output_path)
            print(f"Rendered image saved to {output_path}")

        if vis_keypoints:
            img_array = cv2.imread(os.path.join(img_folder, img_path))
            # img_array = np.asarray(img_pil_nopad)
            for human_idx, human in enumerate(humans):
                # j2d = human['j2d'].cpu().numpy()
                
                # project j3d to 2d
                j3d = human['j3d'].cpu().numpy()
                j2d = j3d @ K[0].cpu().numpy().T
                j2d = j2d[:, :2] / j2d[:, 2:3]

                # # project vertices to 2d
                # v3d = human['v3d'].cpu().numpy()
                # v2d = v3d @ K[0].cpu().numpy().T
                # v2d = v2d[:, :2] / v2d[:, 2:3]

                img_array = visualize_2d_keypoints_in_org_img(img_array, j2d, affine_matrix, human['bbox'], human_idx, human['scores'].detach().cpu().numpy())

            cv2.imwrite(os.path.join(output_dir, f'debug_multihmr_{img_name}_keypoints.jpg'), img_array)

    # convert tensors in the humans_dict to numpy arrays
    for img_name, humans in humans_dict.items():
        for human in humans:
            for key, value in human.items():
                if isinstance(value, torch.Tensor):
                    human[key] = value.cpu().numpy()

    # Save img_size, K, and humans_list as pickle file
    data_to_save = {
        'img_size': img_size,
        'K': K.cpu().numpy(),  # Convert tensor to numpy array
        'humans_dict': humans_dict,
        'affine_matrix_dict': affine_matrix_dict
    }

    # Create a filename with the current timestamp
    # from datetime import datetime
    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f'multihmr_data_{img_folder_name}.pkl'

    # Save the data
    with open(os.path.join(output_dir, filename), 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Data saved to {filename}")

if __name__ == "__main__":
    tyro.cli(main)