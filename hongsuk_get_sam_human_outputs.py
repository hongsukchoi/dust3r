import tyro
import os
import pickle
from collections import defaultdict
from pathlib import Path

def read_multihmr_outputs(multihmr_pkl):
    # Load multihmr data
    with open(multihmr_pkl, 'rb') as f:
        multihmr_data = pickle.load(f)

    return multihmr_data

def read_dust3r_outputs(dust3r_output_path):
    with open(dust3r_output_path, 'rb') as f:
        dust3r_data = pickle.load(f)

    return dust3r_data

def apply_affine_transformation(points, multihmr_affine_matrix, dust3r_affine_matrix):
    # make multihmr_affine_matrix (3,3)
    multihmr_affine_matrix = np.vstack([multihmr_affine_matrix, np.array([0, 0, 1])])
    # make dust3r_affine_matrix (3,3)
    dust3r_affine_matrix = np.vstack([dust3r_affine_matrix, np.array([0, 0, 1])])
    # make homogeneous points (N,3) from (N,2)
    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))]).T

    # apply affine transformation
    points_transformed = np.linalg.inv(dust3r_affine_matrix) @ multihmr_affine_matrix @ points_homogeneous
    points_transformed = points_transformed[:2, :] / points_transformed[2, :]

    points_transformed = points_transformed.T # (J, 2)
    return points_transformed

def visualize_aligned_2d_outputs(score, bbox, j2d_transformed, dust3r_rgbimg):
    # Visualize the keypoints on the original image
    org_img = dust3r_rgbimg.copy() * 255.
    for joint in j2d_transformed:
        org_img = cv2.circle(org_img, (int(joint[0]), int(joint[1])), 3, (0, 255, 0), -1) 

    org_img = cv2.rectangle(org_img, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])), (0, 0, 255), 2)
    org_img = cv2.putText(org_img, f'{score:.2f}', (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return org_img

def align_multihmr_2d_outputs_to_dust3r_2d(multihmr_data, dust3r_data, output_dir, only_main_joints=True, vis=True):

    to_save_data = {}
    for img_name in multihmr_data.keys():
        multihmr_human_list = multihmr_data[img_name]['main_human']
        multihmr_affine_matrix = multihmr_data[img_name]['affine_matrix'] # (2,3)

        # get the rgb image from dust3r data    
        dust3r_rgbimg = dust3r_data[img_name]['rgbimg'] # (H,W,3)
        dust3r_affine_matrix = dust3r_data[img_name]['affine_matrix'] # (2,3)   

        to_save_data[img_name] = {
            'human_det_score': [],
            'human_bbox': [],
            'human_j2d': []
        }

        for multihmr_human in multihmr_human_list:
            # get the 2D human information from multihmr data
            human_det_score = multihmr_human['scores'] # (1,)
            human_bbox = multihmr_human['bbox'] # (4,)
            human_j2d = multihmr_human['j2d'] # (J, 2) 

            # apply affine transformation to the 2D joints and bbox so that it is aligned with the dust3r image
            human_j2d_transformed = apply_affine_transformation(human_j2d, multihmr_affine_matrix, dust3r_affine_matrix)
            # convert bbox from (x,y,w,h) to (x1,y1,x2,y2) of shape (2,2)
            human_bbox = np.array([[human_bbox[0], human_bbox[1]], [human_bbox[0] + human_bbox[2], human_bbox[1] + human_bbox[3]]])
            human_bbox_transformed = apply_affine_transformation(human_bbox, multihmr_affine_matrix, dust3r_affine_matrix)
            # convert back to (x,y,w,h) format
            human_bbox_transformed = np.array([human_bbox_transformed[0][0], human_bbox_transformed[0][1], human_bbox_transformed[1][0] - human_bbox_transformed[0][0], human_bbox_transformed[1][1] - human_bbox_transformed[0][1]])

            to_save_data[img_name]['human_det_score'].append(human_det_score)
            to_save_data[img_name]['human_bbox'].append(human_bbox_transformed)
            if only_main_joints:
                # only keep the body joints excluding face and hand joints
                to_save_data[img_name]['human_j2d'].append(human_j2d_transformed[:get_smplx_joint_names().index('jaw')])
            else:
                to_save_data[img_name]['human_j2d'].append(human_j2d_transformed)

        # visualize the aligned 2D outputs
        if vis:
            aligned_img = dust3r_rgbimg.copy()
            for i in range(len(to_save_data[img_name]['human_det_score'])): 
                aligned_img = visualize_aligned_2d_outputs(to_save_data[img_name]['human_det_score'][i], to_save_data[img_name]['human_bbox'][i], to_save_data[img_name]['human_j2d'][i], aligned_img)
            cv2.imwrite(os.path.join(output_dir, f'{img_name}_human.png'), aligned_img[..., ::-1])

    return to_save_data

def main(output_dir: str = './outputs/egoexo'):
    scene_name = os.path.basename(output_dir)
    multihmr_pkl = Path(output_dir) / f'multihmr_data_{scene_name}.pkl' # '/home/hongsuk/projects/dust3r/outputs/egoexo/multihmr_data_egoexo.pkl'
    dust3r_output_path = Path(output_dir) / f'dust3r_reconstruction_results_{scene_name}.pkl' #'/home/hongsuk/projects/dust3r/outputs/egoexo/dust3r_reconstruction_results_egoexo.pkl'
    output_dir = Path(output_dir) / f'aligned_2d_outputs' #'/home/hongsuk/projects/dust3r/outputs/egoexo/aligned_2d_outputs'

    Path(output_dir).mkdir(parents=True, exist_ok=True) 

    print(f"Reading multihmr data from {multihmr_pkl}")
    multihmr_data = read_multihmr_outputs(multihmr_pkl)
    print(f"Reading dust3r data from {dust3r_output_path}")
    dust3r_data = read_dust3r_outputs(dust3r_output_path)
    print(f"Aligning multihmr 2d outputs to dust3r 2d outputs")
    multihmr_2d_outputs_in_dust3r = align_multihmr_2d_outputs_to_dust3r_2d(multihmr_data, dust3r_data, output_dir, vis=True)








if __name__ == '__main__':
    tyro.cli(main)