import os
import numpy as np
import copy
import pickle
import PIL
import tyro
import tqdm
import pytz
import torch

from datetime import datetime
from pathlib import Path

from multihmr.model import Model

from hongsuk_egohumans_dataloader import create_dataloader

def forward_model(model, input_image, camera_parameters,
                  det_thresh=0.3,
                  nms_kernel_size=1,
                 ):
        
    """ Make a forward pass on an input image and camera parameters. """
    
    # Debug prints to check dtypes
    print("Model dtype:", next(model.parameters()).dtype)
    print("Input image dtype:", input_image.dtype)
    print("Camera parameters dtype:", camera_parameters.dtype)

    # Convert inputs to match model dtype if needed
    model_dtype = next(model.parameters()).dtype
    input_image = input_image.to(model_dtype)
    camera_parameters = camera_parameters.to(model_dtype)
    
    # Forward the model.
    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            humans = model(input_image, 
                           is_training=False, 
                           nms_kernel_size=int(nms_kernel_size),
                           det_thresh=det_thresh,
                           K=camera_parameters)

    return humans

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
    print("Model type:", kwargs['type'], "Image size:", kwargs['img_size'])

    # Load weights into model.
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    
    # Convert model to float32 if needed
    model = model.float()
    
    print("Model dtype after loading:", next(model.parameters()).dtype)
    print("Weights have been loaded")

    return model


def main(output_path: str = './outputs/egohumans'):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # EgoHumans data
    # Fix batch size to 1 for now
    egohumans_data_root = '/home/hongsuk/projects/egohumans/data'
    cam_names = sorted(['cam01', 'cam02', 'cam03', 'cam04'])
    dataset, dataloader = create_dataloader(egohumans_data_root, batch_size=1, split='test', subsample_rate=10, cam_names=cam_names)
    
    # MultiHMR parameters
    device = 'cuda'

    model_path = '/home/hongsuk/projects/SimpleCode/multiview_world/models/multiHMR/multiHMR_896_L.pt'
    det_thresh = 0.3
    nms_kernel_size = 3

    # Load MultiHMR model
    model = load_model(model_path)
    model = model.to(device)

    total_output = {}
    for i, sample in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        x = sample['multihmr_first_cam_input_image'].to(device)
        K = sample['multihmr_intrinsic'].to(device)

        # Make model predictions
        humans = forward_model(model, x, K, det_thresh=det_thresh, nms_kernel_size=nms_kernel_size)
        # len(humans) is the number of humans in the image
        # humans[i].keys: ['scores', 'loc', 'transl', 'transl_pelvis', 'rotvec', 'expression', 'shape', 'v3d', 'j3d', 'j2d']
        
        # Save output
        output_name = f"{sample['sequence'][0]}_{sample['frame'][0].item()}_{''.join(cam_names)}"

        total_output[output_name] = {
            'first_cam_humans': humans,
        }
        break


    # Save total output
    # get date and time (day:hour:minute)
    # time in pacific time
    now = datetime.now(pytz.timezone('US/Pacific')).strftime("%d:%H:%M")
    with open(os.path.join(output_path, f'multihmr_output_{now}.pkl'), 'wb') as f:
        pickle.dump(total_output, f)

if __name__ == '__main__':
    tyro.cli(main)