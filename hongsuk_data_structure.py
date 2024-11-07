
"""
Directory structure of the egohumans vitpose hmr2 hamer output:

egohumans_vitpose_hmr2_hamer_output_dir: '/scratch/one_month/2024_10/lmueller/egohuman/camera_ready'

${egohumans_vitpose_hmr2_hamer_output_dir}/
    01_tagging/
        001_tagging/
            processed_data/humanwithhand/
                cam01/
                    identified_predictions/
                        __hongsuk_identified_vitpose_bbox_smplx_frame{05d:idx}.pkl # 1-indexed
                        ...
                ...
        ....
    ...

Data structure of the __hongsuk_identified_vitpose_bbox_smplx_frame{05d:idx}.pkl:
# The corresponding image can be found here: /scratch/partial_datasets/bedlam/egohumans/extracted/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready
# Ex) /scratch/partial_datasets/bedlam/egohumans/extracted/media/rawalk/disk1/rawalk/datasets/ego_exo/camera_ready/01_tagging/001_tagging/exo/cam01/images/00001.jpg
# Beaware that sometimes the confidence can be just 0 even though the bbox and pose2d are good enough
# The joint order of ViTPose is following the COCO-WholeBody format: https://github.com/open-mmlab/mmhuman3d/blob/main/mmhuman3d/core/conventions/keypoints_mapping/coco_wholebody.py
# Some humans may no exist even though you can obvisouly see them in the image :) This is due to failtures of assigning ids to the humans.
{
    'aria01':
        'bbox': (5,) [x, y, w, h, confidence]
        'pose2d': (133,3) x,y, confidence
        'params': dictionary of SMPL-X parameters
        'original_index_in_mono_vitpose_pred': list of indices in the original mono prediction file
    'aria02':
        ...
}
"""