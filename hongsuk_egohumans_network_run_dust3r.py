import os
import numpy as np
import copy
import pickle
import PIL
import tyro
from pathlib import Path

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images

from hongsuk_egohumans_dataloader import create_dataloader


def main():
    
    # EgoHumans data
    # Fix batch size to 1 for now
    egohumans_data_root = '/home/hongsuk/projects/egohumans/data'
    cam_names = ['cam01', 'cam02', 'cam03', 'cam04']
    dataloader = create_dataloader(egohumans_data_root, batch_size=1, split='test', subsample_rate=10, cam_names=cam_names)

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

    total_output = [] 
    for i, sample in enumerate(dataloader):
        imgs = [sample['multiview_images'][cam_name] for cam_name in cam_names]
        # # squeeze the batch dimension of 'img' and 'true_shape'
        # for img in imgs:
        #     img['img'] = img['img'][0]
        #     img['true_shape'] = img['true_shape'][0]

        affine_transforms = [sample['multiview_affine_transforms'][cam_name] for cam_name in cam_names]

        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1
        pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
        print(f"Running DUSt3R network inference with {len(imgs)} images, scene graph type: {scenegraph_type}, {len(pairs)} pairs")
        output = inference(pairs, model, device, batch_size=1, verbose=not silent)

        total_output.append({
            'affine_matrices': affine_transforms,
            'output': output,
            'img_names': (sample['sequence'], sample['frame'].tolist(), cam_names)
        })


if __name__ == '__main__':
    tyro.cli(main)