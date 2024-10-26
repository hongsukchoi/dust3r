import math
import gradio
import os
import os.path as osp
import numpy as np
import torch
import functools
import trimesh
import copy
import pickle
from scipy.spatial.transform import Rotation
import tempfile
import shutil
import tyro
import PIL
import cv2
from pathlib import Path
from dust3r_config import get_dust3r_config

from dust3r.inference import inference
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images, rgb
from dust3r.utils.device import to_numpy
from dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode

from hongsuk_vis_viser_env_only import show_env_in_viser
from hongsuk_vis_viser_env_human import show_env_human_in_viser
from multihmr.blocks import SMPL_Layer

# hard coding to get the affine transform matrix
def preprocess_and_get_transform(file, size=512, square_ok=False):
    img = PIL.Image.open(file)
    original_width, original_height = img.size
    
    # Step 1: Resize
    S = max(img.size)
    if S > size:
        interp = PIL.Image.LANCZOS
    else:
        interp = PIL.Image.BICUBIC
    new_size = tuple(int(round(x*size/S)) for x in img.size)
    img_resized = img.resize(new_size, interp)
    
    # Calculate center of the resized image
    cx, cy = img_resized.size[0] // 2, img_resized.size[1] // 2
    
    # Step 2: Crop
    halfw, halfh = ((2*cx)//16)*8, ((2*cy)//16)*8
    if not square_ok and new_size[0] == new_size[1]:
        halfh = 3*halfw//4
    
    img_cropped = img_resized.crop((cx-halfw, cy-halfh, cx+halfw, cy+halfh))
    
    # Calculate the total transformation
    scale_x = new_size[0] / original_width
    scale_y = new_size[1] / original_height
    
    translate_x = (cx - halfw) / scale_x
    translate_y = (cy - halfh) / scale_y
    
    affine_matrix = np.array([
        [1/scale_x, 0, translate_x],
        [0, 1/scale_y, translate_y]
    ])
    
    return img_cropped, affine_matrix


def global_alignment_optimization(scene, device, silent, niter, schedule, lr=0.01, init='mst', pts3d=None, im_focals=None, im_poses=None, smplx_3d_params=None, smplx_2d_data=None, output_dir=None):
    lr = 0.01
    if init == 'mst':
        loss = scene.compute_global_alignment(init=init, niter=niter, schedule=schedule, lr=lr)
    elif init == 'known_pts3d':
        scene.norm_pw_scale = False
        loss = scene.compute_global_alignment(init=init, niter=niter, schedule=schedule, lr=lr, pts3d=pts3d, im_focals=im_focals, im_poses=im_poses)
    elif init == 'known_pts3d_and_smplx':
        scene.norm_pw_scale = False
        loss = scene.compute_global_alignment(init=init, niter=niter, schedule=schedule, lr=lr, pts3d=pts3d, im_focals=im_focals, im_poses=im_poses,
                                            smplx_3d_params=smplx_3d_params, smplx_2d_data=smplx_2d_data, output_dir=output_dir)   
    print('final loss: ', loss)

    # get optimized values from scene
    # 3D pointcloud from depthmap, poses and intrinsics
    # pts3d: list of N pointclouds, each shape(H, W,3)
    # confs: list of N confidence scores, each shape(H, W)
    # msk: boolean mask of valid points, shape(H, W)
    pts3d = scene.get_pts3d()
    depths = scene.get_depthmaps()
    msk = scene.get_masks()
    confs = [c for c in scene.im_conf]
    intrinsics = scene.get_intrinsics() # N intrinsics # (N, 3, 3)
    cams2world = scene.get_im_poses() # (N,4,4)

    return scene, intrinsics, cams2world, pts3d, depths, msk, confs

def get_reconstructed_scene(outdir, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid, run_inference=True, loaded_optimized_output=None, model=None, output_dir=None):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    
    # get affine transform matrix list
    affine_matrix_list = []
    for file in filelist:
        img_cropped, affine_matrix = preprocess_and_get_transform(file)
        affine_matrix_list.append(affine_matrix)

    # run dust3r inference with the given model
    print("File list: ", filelist)
    if run_inference:
        imgs = load_images(filelist, size=image_size, verbose=not silent)
        if len(imgs) == 1:
            imgs = [imgs[0], copy.deepcopy(imgs[0])]
            imgs[1]['idx'] = 1
        if scenegraph_type == "swin":
            scenegraph_type = scenegraph_type + "-" + str(winsize)
        elif scenegraph_type == "oneref":
            scenegraph_type = scenegraph_type + "-" + str(refid)

        pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=1, verbose=not silent)

        # Select an optimizer
        mode = GlobalAlignerMode.PointCloudOptimizer if len(filelist) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
        
        # Do the optimization
        scene, intrinsics, cams2world, pts3d, depths, msk, confs = global_alignment_optimization(scene, device, silent, niter, schedule, lr=0.01, init='mst')
        # print("After optimization, im_focals: ", scene.im_focals)
        # print("After optimization, im_poses: ", scene.im_poses)

    else:
        if loaded_optimized_output is None:
            raise ValueError('loaded_optimized_output is None, please provide the output from dust3r inference')
        else:
            output = loaded_optimized_output['non_img_specific']['raw_network_prediction']

            # Select an optimizer
            mode = GlobalAlignerMode.PointCloudOptimizer if len(filelist) > 2 else GlobalAlignerMode.PairViewer
            scene = global_aligner(output, device=device, mode=mode, verbose=not silent, has_human_cue=True)

            # load the precomputed 3D points, camera poses, and intrinsics, and the human data
            pts3d = []
            im_poses_4by4 = []
            unnormalized_im_focals = []

            smplx_3d_params = []
            smplx_2d_data = []
            for img_idx, img_name in enumerate(sorted(loaded_optimized_output.keys())):
                if img_name == 'non_img_specific':
                    continue
                print(f"Processing {img_name}")
                file_name = osp.basename(filelist[img_idx]).split('.')[0]
                assert file_name == img_name, f"File name {file_name} does not match {img_name}"

                # make them to torch tensors
                pts3d.append(torch.from_numpy(loaded_optimized_output[img_name]['pts3d']).to(device))
                im_poses_4by4.append(torch.from_numpy(loaded_optimized_output[img_name]['cam2world']).to(device))
                unnormalized_im_focals.append(loaded_optimized_output[img_name]['intrinsic'][0][0])

                # load the smplx parameters that are to be optimized
                smplx_3d_params.append(loaded_optimized_output[img_name]['multihmr_3d_outputs']) # dictionary of transl, rotvec, shape, expression

                # load the 2D joints and bboxes that will be ground truth for the 2D supervision
                smplx_2d_data.append(loaded_optimized_output[img_name]['multihmr_2d_outputs']) # dictionary of human_bbox, human_j2d, human_det_score 
            im_poses_4by4 = torch.stack(im_poses_4by4)
            # just pick one view smplx_3d_params for initialization
            smplx_3d_params = smplx_3d_params[0]

            scene, intrinsics, cams2world, pts3d, depths, msk, confs = \
                global_alignment_optimization(scene, device, silent, niter, schedule, lr=0.01, init='known_pts3d_and_smplx', \
                                            pts3d=pts3d, im_focals=unnormalized_im_focals, im_poses=im_poses_4by4,
                                            smplx_3d_params=smplx_3d_params, smplx_2d_data=smplx_2d_data, output_dir=output_dir)
            # print("After optimization, human_transl: ", scene.human_transl)
            # print("After optimization, smplx_params: ", scene.get_smplx_params())


    intrinsics = to_numpy(intrinsics)
    cams2world = to_numpy(cams2world)
    pts3d = to_numpy(pts3d) # pts3d: list of N pointclouds, each shape(H, W,3)
    depths = to_numpy(depths) # depths: list of N depthmaps, each shape(H, W)
    msk = to_numpy(msk) # boolean mask of valid points, shape(H, W)
    confs = to_numpy(confs) # list of N confidence scores, each shape(H, W)
    rgbimg = scene.imgs   # list of N numpy images with shape (H,W,3) , rgb

    smplx_params = scene.get_smplx_params()
    return output, rgbimg, intrinsics, cams2world, pts3d, depths, msk, confs, affine_matrix_list, smplx_params


def main(run_dust3r: bool = False, config_yaml: str = 'dust3r_config.yaml', model_path: str = '/home/hongsuk/projects/SimpleCode/multiview_world/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', out_dir: str = './outputs', img_dir: str = './images'):
    # Get the config
    config = get_dust3r_config(config_yaml)
    if 'monst3r' in model_path.lower(): 
        config.update_filelist(img_dir, start_frame=0, end_frame=100, stride=5)
    else:
        config.update_filelist(img_dir)

    # Add any custom arguments here
    custom_args = {
        'run_inference': run_dust3r,
        'loaded_optimized_output': None
    }
    if img_dir.endswith('/'):
        img_dir = img_dir[:-1]
    out_dir = osp.join(out_dir, osp.basename(img_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model_name = osp.basename(model_path).split('_')[0].lower()
    if run_dust3r:
        # Load your model here
        from dust3r.model import AsymmetricCroCo3DStereo
        model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(config.device)
        output_file = osp.join(out_dir, f'{model_name}_reconstruction_results_{osp.basename(img_dir)}.pkl')
    else:
        model = None
        input_file = f'/home/hongsuk/projects/dust3r/outputs/{osp.basename(img_dir)}/rescaled_dust3r_reconstruction_results_{osp.basename(img_dir)}_with_multihmr_aligned.pkl'
        with open(input_file, 'rb') as f:
            optimized_output = pickle.load(f)
        print(f"Loaded optimized output from {input_file}")
        custom_args['loaded_optimized_output'] = optimized_output
        output_file = osp.join(out_dir, f'{model_name}_reconstruction_results_reoptimized_{osp.basename(img_dir)}.pkl')
        print(f"Reoptimized output will be saved to {output_file}")
    custom_args['model'] = model
    custom_args['output_dir'] = out_dir

    # Create a lambda function to wrap get_reconstructed_scene
    get_reconstructed_scene_lambda = lambda *args, **kwargs: get_reconstructed_scene(*args, **kwargs)
    # Combine default parameters with custom arguments
    params = config.get_default_params()
    combined_args = {**dict(zip(['outdir', 'device', 'silent', 'image_size', 'filelist', 'schedule', 'niter', 'min_conf_thr', 'as_pointcloud', 'mask_sky', 'clean_depth', 'transparent_cams', 'cam_size', 'scenegraph_type', 'winsize', 'refid'], params)), **custom_args}
    # Call the lambda function with combined arguments
    output, rgbimg, intrinsics, cams2world, pts3d, depths, msk, confs, affine_matrix_list, smplx_params = get_reconstructed_scene_lambda(**combined_args)
    
    # Save the results as a pickle file
    results = {}
    non_img_specific_results = {}
    non_img_specific_results['raw_network_prediction'] = output
    non_img_specific_results['smplx_params'] = smplx_params
    results['non_img_specific'] = non_img_specific_results
    for i, f in enumerate(config.filelist):
        img_name = osp.basename(f).split('.')[0]
        results[img_name] = {
            'rgbimg': rgbimg[i],
            'intrinsic': intrinsics[i],
            'cam2world': cams2world[i],
            'pts3d': pts3d[i],
            'depths': depths[i],
            'msk': msk[i],
            'conf': confs[i],
            'affine_matrix': affine_matrix_list[i]
        }
    
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to {output_file}")

    if not run_dust3r:
        # get vertices and faces from smplx_params
        smplx_layer = SMPL_Layer(type='smplx', gender='neutral', num_betas=10, kid=False, person_center='head')
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
        smplx_vertices_world = {
            'world': smplx_vertices.detach().squeeze().cpu().numpy()
        }
        smplx_faces = smplx_layer.bm_x.faces

        show_env_human_in_viser(output_file, world_scale_factor=1., smplx_vertices_dict=smplx_vertices_world, smplx_faces=smplx_faces)
    else:
        show_env_in_viser(output_file, world_scale_factor=5.)

if __name__ == '__main__':
    tyro.cli(main)