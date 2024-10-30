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

def transform_keypoints(homogeneous_keypoints: np.ndarray, affine_matrix: np.ndarray):
    # Ensure keypoints is a numpy array
    homogeneous_keypoints = np.array(homogeneous_keypoints)

    # Apply the transformation
    transformed_keypoints = np.dot(affine_matrix, homogeneous_keypoints.T).T
    
    # Round to nearest integer for pixel coordinates
    transformed_keypoints = np.round(transformed_keypoints).astype(int)
    
    return transformed_keypoints

def check_affine_matrix(test_img: PIL.Image, original_image: PIL.Image, affine_matrix: np.ndarray):
    assert affine_matrix.shape == (2, 3)

    # get pixels near the center of the image in the new image space
    # Sample 100 pixels near the center of the image
    w, h = test_img.size
    center_x, center_y = w // 2, h // 2
    radius = min(w, h) // 4  # Use a quarter of the smaller dimension as the radius

    # Generate random offsets within the circular region
    num_samples = 100
    theta = np.random.uniform(0, 2*np.pi, num_samples)
    r = np.random.uniform(0, radius, num_samples)
    
    # Convert polar coordinates to Cartesian
    x_offsets = r * np.cos(theta)
    y_offsets = r * np.sin(theta)
    
    # Add offsets to the center coordinates and ensure they're within image bounds
    sampled_x = np.clip(center_x + x_offsets, 0, w-1).astype(int)
    sampled_y = np.clip(center_y + y_offsets, 0, h-1).astype(int)
    
    # Create homogeneous coordinates
    pixels_near_center = np.column_stack((sampled_x, sampled_y, np.ones(num_samples)))
    
    # draw the pixels on the image and save it
    test_img_pixels = np.asarray(test_img).copy().astype(np.uint8)
    for x, y in zip(sampled_x, sampled_y):
        test_img_pixels = cv2.circle(test_img_pixels, (x, y), 3, (0, 255, 0), -1)
    PIL.Image.fromarray(test_img_pixels).save('test_new_img_pixels.png')

    transformed_keypoints = transform_keypoints(pixels_near_center, affine_matrix)
    # Load the original image
    original_img_array = np.array(original_image)

    # Draw the transformed keypoints on the original image
    for point in transformed_keypoints:
        x, y = point[:2]
        # Ensure the coordinates are within the image bounds
        if 0 <= x < original_image.width and 0 <= y < original_image.height:
            cv2.circle(original_img_array, (int(x), int(y)), int(3*affine_matrix[0,0]), (255, 0, 0), -1)

    # Save the image with drawn keypoints
    PIL.Image.fromarray(original_img_array).save('test_original_img_keypoints.png')

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


def global_alignment_optimization(output, imgs, device, silent, niter, schedule, lr=0.01, init='mst', scene=None, pts3d=None, im_focals=None, im_poses=None):
    if scene is None:
        mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer
        scene = global_aligner(output, device=device, mode=mode, verbose=not silent)
    else:
        mode = GlobalAlignerMode.PointCloudOptimizer if len(imgs) > 2 else GlobalAlignerMode.PairViewer

    lr = 0.01
    if mode == GlobalAlignerMode.PointCloudOptimizer:
        if init == 'mst':
            loss = scene.compute_global_alignment(init=init, niter=niter, schedule=schedule, lr=lr)
        elif init == 'known_pts3d':
            loss = scene.compute_global_alignment(init=init, niter=niter, schedule=schedule, lr=lr, pts3d=pts3d, im_focals=im_focals, im_poses=im_poses)
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

def get_reconstructed_scene(outdir, model, device, silent, image_size, filelist, schedule, niter, min_conf_thr,
                            as_pointcloud, mask_sky, clean_depth, transparent_cams, cam_size,
                            scenegraph_type, winsize, refid):
    """
    from a list of images, run dust3r inference, global aligner.
    then run get_3D_model_from_scene
    """
    imgs = load_images(filelist, size=image_size, verbose=not silent)

    # get affine transform matrix list
    affine_matrix_list = []
    # img_cropped_list = []
    for file in filelist:
        img_cropped, affine_matrix = preprocess_and_get_transform(file)
        affine_matrix_list.append(affine_matrix)
        # img_cropped_list.append(img_cropped)

    # CHECK the first image
    # test_img = img_cropped_list[0]
    # org_img = PIL.Image.open(filelist[0])
    # check_affine_matrix(test_img, org_img, affine_matrix_list[0])
    # import pdb; pdb.set_trace()


    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    output = inference(pairs, model, device, batch_size=1, verbose=not silent)
    # Comments from Dust3r README.md
    # here, view1, pred1, view2, pred2 are dicts of lists of len(2)
    #  -> because we symmetrize we have (im1, im2) and (im2, im1) pairs
    # in each view you have:
    # an integer image identifier: view1['idx'] and view2['idx']
    # the img: view1['img'] and view2['img']
    # the image shape: view1['true_shape'] and view2['true_shape']
    # an instance string output by the dataloader: view1['instance'] and view2['instance']
    # pred1 and pred2 contains the confidence values: pred1['conf'] and pred2['conf']
    # pred1 contains 3D points for view1['img'] in view1['img'] space: pred1['pts3d']
    # pred2 contains 3D points for view2['img'] in view1['img'] space: pred2['pts3d_in_other_view']

    scene, intrinsics, cams2world, pts3d, depths, msk, confs = global_alignment_optimization(output, imgs, device, silent, niter, schedule, lr=0.01, init='mst')

    redo_global_alignment = True
    if redo_global_alignment:
        # do the optimization again with scaled 3D points and camera poses
        scale = 10.
        pts3d_scaled = [p * scale for p in pts3d]
        cams2world_scaled = []
        for c in range(len(cams2world)):
            cams2world[c][:3, 3] *= scale
            cams2world_scaled.append(cams2world[c])
        im_poses = torch.stack(cams2world_scaled)
        im_focals = [None] * len(imgs) # to_numpy(scene.im_focals).tolist()
        # print("Check im_focals: ", scene.im_focals)
        # TEMP
        scene.norm_pw_scale = False
        print("Redoing global alignment with scaled 3D points and camera poses; scale factor: ", scale)
        scene, intrinsics, cams2world, pts3d, depths, msk, confs = \
        global_alignment_optimization(output, imgs, device, silent, niter, schedule, lr=0.01, init='known_pts3d', \
                                      scene=scene, pts3d=pts3d_scaled, im_focals=im_focals, im_poses=im_poses)
        # import pdb; pdb.set_trace()

    # 3D pointcloud from depthmap, poses and intrinsics
    # pts3d: list of N pointclouds, each shape(H, W,3)
    # confs: list of N confidence scores, each shape(H, W)
    # msk: boolean mask of valid points, shape(H, W)
    intrinsics = to_numpy(intrinsics)
    cams2world = to_numpy(cams2world)
    pts3d = to_numpy(pts3d)
    depths = to_numpy(depths)
    msk = to_numpy(msk)
    confs = to_numpy(confs)

    # Hongsuk TEMP
    im_poses = to_numpy(scene.im_poses)
    im_focals = to_numpy(scene.im_focals)

    rgbimg = scene.imgs   # list of N numpy images with shape (H,W,3) , rgb

    return rgbimg, intrinsics, cams2world, pts3d, depths, msk, confs, affine_matrix_list


def main(config_yaml: str = 'dust3r_config.yaml', model_path: str = '/home/hongsuk/projects/SimpleCode/multiview_world/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', out_dir: str = './outputs', img_dir: str = './images'):
    config = get_dust3r_config(config_yaml)
    # Temp hard coding - hongsuk
    if 'monst3r' in model_path.lower(): 
        config.update_filelist(img_dir, start_frame=0, end_frame=100, stride=5)
    else:
        config.update_filelist(img_dir)

    # Load your model here
    from dust3r.model import AsymmetricCroCo3DStereo
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(config.device)
    config.model = model

    params = config.get_default_params()
    rgbimg, intrinsics, cams2world, pts3d, depths, msk, confs, affine_matrix_list = get_reconstructed_scene(*params)
    
    # Save the results as a pickle file
    results = {}
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

    # Add timestamp to the filename
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if img_dir.endswith('/'):
        img_dir = img_dir[:-1]
    out_dir = osp.join(out_dir, osp.basename(img_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model_name = osp.basename(model_path).split('_')[0].lower()
    output_file = osp.join(out_dir, f'{model_name}_reconstruction_results_{osp.basename(img_dir)}_{timestamp}.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to {output_file}")

    show_env_in_viser(output_file, world_scale_factor=5.)


if __name__ == '__main__':
    tyro.cli(main)