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



# get the affine transform matrix from cropped image to original image
def get_affine_transform(file, size=512, square_ok=False):
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


def run_dust3r_network_inference(filelist, image_size, device, model, scenegraph_type, winsize, refid, silent=False):
    imgs = load_images(filelist, size=image_size, verbose=not silent)
    if len(imgs) == 1:
        imgs = [imgs[0], copy.deepcopy(imgs[0])]
        imgs[1]['idx'] = 1
    if scenegraph_type == "swin":
        scenegraph_type = scenegraph_type + "-" + str(winsize)
    elif scenegraph_type == "oneref":
        scenegraph_type = scenegraph_type + "-" + str(refid)

    pairs = make_pairs(imgs, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
    print(f"Running DUSt3R network inference with {len(filelist)} images, scene graph type: {scenegraph_type}, {len(pairs)} pairs")
    output = inference(pairs, model, device, batch_size=1, verbose=not silent)

    return output


def main(model_path: str = '/home/hongsuk/projects/SimpleCode/multiview_world/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth', out_dir: str = './outputs', img_dir: str = './images'):    
    # Get the filelist
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    filelist = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir)
                         if f.lower().endswith(image_extensions)])
    print("Input file list: ", filelist)

    # Default parameters
    image_size = 512
    device = 'cuda'
    silent = False
    # Config for the scene graph that makes pair inputs
    scenegraph_type = 'complete'
    winsize = 1
    refid = 0

    # Load your model here
    from dust3r.model import AsymmetricCroCo3DStereo
    model = AsymmetricCroCo3DStereo.from_pretrained(model_path).to(device)

    # Run the dust3r network inference
    output = run_dust3r_network_inference(filelist, image_size, device, model, scenegraph_type, winsize, refid, silent)

    # get the affine transform matrix from cropped image to original image
    affine_matrices = []
    for file in filelist:
        affine_matrix = get_affine_transform(file, image_size)
        affine_matrices.append(affine_matrix)

    # Save the output
    if img_dir.endswith('/'):
        img_dir = img_dir[:-1]
    out_dir = os.path.join(out_dir, os.path.basename(img_dir))
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    output_file = os.path.join(out_dir, f'dust3r_network_output_pointmaps_{os.path.basename(img_dir)}.pkl')
    total_output = {    
        'output': output,
        'affine_matrices': affine_matrices,
        'img_names': [os.path.basename(file).split('.')[0] for file in filelist]
    }
    with open(output_file, 'wb') as f:
        pickle.dump(total_output, f)
    print(f"Saved DUSt3R network output and affine matrices to {output_file}")

if __name__ == '__main__':
    tyro.cli(main)