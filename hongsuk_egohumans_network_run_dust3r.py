import os
import numpy as np
import copy
import pickle
import PIL
import tyro
import tqdm
import pytz

from datetime import datetime
from pathlib import Path

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images

from hongsuk_egohumans_dataloader import create_dataloader


def main(output_path: str = './outputs/egohumans'):
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # EgoHumans data
    # Fix batch size to 1 for now
    egohumans_data_root = '/home/hongsuk/projects/egohumans/data'
    cam_names = sorted(['cam01', 'cam02', 'cam03', 'cam04'])
    dataset, dataloader = create_dataloader(egohumans_data_root, batch_size=1, split='test', subsample_rate=10, cam_names=cam_names)

    # Dust3r parameters
    device = 'cuda'
    silent = False
    scenegraph_type = 'complete'
    winsize = 1
    refid = 0
    model_path = '/home/hongsuk/projects/SimpleCode/multiview_world/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth'

    # get dust3r network model
    from dust3r.model import AsymmetricCroCo3DStereo
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

    total_output = {}
    for i, sample in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
        imgs = [sample['multiview_images'][cam_name] for cam_name in cam_names]

        # for single view, make a pair with itself
        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1
        pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
        print(f"Running DUSt3R network inference with {len(imgs)} images, scene graph type: {scenegraph_type}, {len(pairs)} pairs")
        output = inference(pairs, model, device, batch_size=1, verbose=not silent)

        # Save output
        output_name = f"{sample['sequence'][0]}_{sample['frame'][0].item()}_{''.join(cam_names)}"
        # dust3r input image to original image transform; to be compatible with the outputs from different methods that use different input image sizes
        affine_transforms = [sample['multiview_affine_transforms'][cam_name] for cam_name in cam_names]
        total_output[output_name] = {
            'affine_matrices': affine_transforms,
            'output': output,
            'img_names': cam_names
        }


    # Save total output
    # get date and time (day:hour:minute)
    # time in pacific time
    now = datetime.now(pytz.timezone('US/Pacific')).strftime("%d:%H:%M")
    with open(os.path.join(output_path, f'dust3r_network_output_{now}.pkl'), 'wb') as f:
        pickle.dump(total_output, f)

if __name__ == '__main__':
    tyro.cli(main)