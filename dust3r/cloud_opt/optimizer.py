# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# Main class for the implementation of the global alignment
# --------------------------------------------------------
import numpy as np
import torch
import torch.nn as nn
import cv2
import os.path as osp

from dust3r.cloud_opt.base_opt import BasePCOptimizer
from dust3r.utils.geometry import xy_grid, geotrf, inv
from dust3r.utils.device import to_cpu, to_numpy

from dust3r.cloud_opt.init_im_poses import init_minimum_spanning_tree, rigid_points_registration, fast_pnp
from dust3r.cloud_opt.commons import edge_str, i_j_ij, compute_edge_scores

class PointCloudOptimizer(BasePCOptimizer):
    """ Optimize a global scene, given a list of pairwise observations.
    Graph node: images
    Graph edges: observations = (pred1, pred2)
    """

    def __init__(self, *args, has_human_cue=False, optimize_pp=False, focal_break=20, **kwargs):
        super().__init__(*args, **kwargs)

        self.has_im_poses = True  # by definition of this class
        self.focal_break = focal_break

        # Hongsuk added
        self.has_human_cue = has_human_cue
        if self.has_human_cue:
            from multihmr.blocks import SMPL_Layer
            from multihmr.utils import get_smplx_joint_names
            self.human_loss_weight = 15.0
            self.smplx_layer = SMPL_Layer(type='smplx', gender='neutral', num_betas=10, kid=False, person_center='head')

            # make smplx_layer.bm_x attributes all requires.grad False
            # Note that self.human_transl and self.human_global_rotvec are independent from smplx_layer.bm_x according to MultiHMR code
            for attr in ['betas', 'global_orient', 'body_pose', 'transl', 'left_hand_pose', 'right_hand_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'expression']:
                # self.smplx_layer.bm_x.__setattr__(attr).requires_grad_(False)
                getattr(self.smplx_layer.bm_x, attr).requires_grad_(False)

            # initialize with random values
            # at the moment, only single person
            # these are things to optimize! - Hongsuk
            self.human_transl = nn.Parameter(torch.randn(1, 3).to(self.device)) # (1, 3)
            self.human_global_rotvec = nn.Parameter(torch.randn(1, 1, 3).to(self.device)) # (1, 1, 3)
            self.human_relative_rotvec = nn.Parameter(torch.randn(1, 52, 3).to(self.device)) # (1, 52, 3)
            self.human_shape = nn.Parameter(torch.randn(1, 10).to(self.device)) # (1, 10)
            self.human_expression = nn.Parameter(torch.randn(1, 10).to(self.device)) # (1, 10)

            # set human_relative_rotvec, human_shape, human_expression requires_grad to False
            # self.human_relative_rotvec.requires_grad_(False)
            # self.human_shape.requires_grad_(False)
            self.human_expression.requires_grad_(False)

        # adding thing to optimize
        self.im_depthmaps = nn.ParameterList(torch.randn(H, W)/10-3 for H, W in self.imshapes)  # log(depth)
        self.im_poses = nn.ParameterList(self.rand_pose(self.POSE_DIM) for _ in range(self.n_imgs))  # camera poses
        self.im_focals = nn.ParameterList(torch.FloatTensor(
            [self.focal_break*np.log(max(H, W))]) for H, W in self.imshapes)  # camera intrinsics
        self.im_pp = nn.ParameterList(torch.zeros((2,)) for _ in range(self.n_imgs))  # camera intrinsics
        self.im_pp.requires_grad_(optimize_pp)

        self.imshape = self.imshapes[0]
        im_areas = [h*w for h, w in self.imshapes]
        self.max_area = max(im_areas)

        # adding thing to optimize
        self.im_depthmaps = ParameterStack(self.im_depthmaps, is_param=True, fill=self.max_area)
        self.im_poses = ParameterStack(self.im_poses, is_param=True)
        self.im_focals = ParameterStack(self.im_focals, is_param=True)
        self.im_pp = ParameterStack(self.im_pp, is_param=True)
        self.register_buffer('_pp', torch.tensor([(w/2, h/2) for h, w in self.imshapes]))
        self.register_buffer('_grid', ParameterStack(
            [xy_grid(W, H, device=self.device) for H, W in self.imshapes], fill=self.max_area))

        # pre-compute pixel weights
        self.register_buffer('_weight_i', ParameterStack(
            [self.conf_trf(self.conf_i[i_j]) for i_j in self.str_edges], fill=self.max_area))
        self.register_buffer('_weight_j', ParameterStack(
            [self.conf_trf(self.conf_j[i_j]) for i_j in self.str_edges], fill=self.max_area))

        # precompute aa
        self.register_buffer('_stacked_pred_i', ParameterStack(self.pred_i, self.str_edges, fill=self.max_area))
        self.register_buffer('_stacked_pred_j', ParameterStack(self.pred_j, self.str_edges, fill=self.max_area))
        self.register_buffer('_ei', torch.tensor([i for i, j in self.edges]))
        self.register_buffer('_ej', torch.tensor([j for i, j in self.edges]))
        self.total_area_i = sum([im_areas[i] for i, j in self.edges])
        self.total_area_j = sum([im_areas[j] for i, j in self.edges])
        print("total_area_i: ", self.total_area_i)
        print("total_area_j: ", self.total_area_j)

    def _check_all_imgs_are_selected(self, msk):
        assert np.all(self._get_msk_indices(msk) == np.arange(self.n_imgs)), 'incomplete mask!'

    def preset_pose(self, known_poses, pose_msk=None):  # cam-to-world
        self._check_all_imgs_are_selected(pose_msk)

        if isinstance(known_poses, torch.Tensor) and known_poses.ndim == 2:
            known_poses = [known_poses]
        for idx, pose in zip(self._get_msk_indices(pose_msk), known_poses):
            if self.verbose:
                print(f' (setting pose #{idx} = {pose[:3,3]})')
            self._no_grad(self._set_pose(self.im_poses, idx, torch.tensor(pose)))

        # normalize scale if there's less than 1 known pose
        n_known_poses = sum((p.requires_grad is False) for p in self.im_poses)
        self.norm_pw_scale = (n_known_poses <= 1)

        self.im_poses.requires_grad_(False)
        self.norm_pw_scale = False

    def preset_focal(self, known_focals, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, focal in zip(self._get_msk_indices(msk), known_focals):
            if self.verbose:
                print(f' (setting focal #{idx} = {focal})')
            self._no_grad(self._set_focal(idx, focal))

        self.im_focals.requires_grad_(False)

    def preset_principal_point(self, known_pp, msk=None):
        self._check_all_imgs_are_selected(msk)

        for idx, pp in zip(self._get_msk_indices(msk), known_pp):
            if self.verbose:
                print(f' (setting principal point #{idx} = {pp})')
            self._no_grad(self._set_principal_point(idx, pp))

        self.im_pp.requires_grad_(False)

    def _get_msk_indices(self, msk):
        if msk is None:
            return range(self.n_imgs)
        elif isinstance(msk, int):
            return [msk]
        elif isinstance(msk, (tuple, list)):
            return self._get_msk_indices(np.array(msk))
        elif msk.dtype in (bool, torch.bool, np.bool_):
            assert len(msk) == self.n_imgs
            return np.where(msk)[0]
        elif np.issubdtype(msk.dtype, np.integer):
            return msk
        else:
            raise ValueError(f'bad {msk=}')

    def _no_grad(self, tensor):
        assert tensor.requires_grad, 'it must be True at this point, otherwise no modification occurs'

    def _set_focal(self, idx, focal, force=False):
        param = self.im_focals[idx]
        # if param.requires_grad or force:  # can only init a parameter not already initialized
        #     param.data[:] = self.focal_break * np.log(focal)
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = torch.Tensor([self.focal_break * np.log(focal)])

        return param

    def get_focals(self):
        log_focals = torch.stack(list(self.im_focals), dim=0)
        return (log_focals / self.focal_break).exp()

    def get_known_focal_mask(self):
        return torch.tensor([not (p.requires_grad) for p in self.im_focals])

    def _set_principal_point(self, idx, pp, force=False):
        param = self.im_pp[idx]
        H, W = self.imshapes[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = to_cpu(to_numpy(pp) - (W/2, H/2)) / 10
        return param

    def get_principal_points(self):
        return self._pp + 10 * self.im_pp

    def get_intrinsics(self):
        K = torch.zeros((self.n_imgs, 3, 3), device=self.device)
        focals = self.get_focals().flatten()
        K[:, 0, 0] = K[:, 1, 1] = focals
        K[:, :2, 2] = self.get_principal_points()
        K[:, 2, 2] = 1
        return K

    def get_im_poses(self):  # cam to world
        cam2world = self._get_poses(self.im_poses)
        return cam2world

    def _set_depthmap(self, idx, depth, force=False):
        depth = _ravel_hw(depth, self.max_area)

        param = self.im_depthmaps[idx]
        if param.requires_grad or force:  # can only init a parameter not already initialized
            param.data[:] = depth.log().nan_to_num(neginf=0)
        return param

    def get_depthmaps(self, raw=False):
        res = self.im_depthmaps.exp()
        if not raw:
            res = [dm[:h*w].view(h, w) for dm, (h, w) in zip(res, self.imshapes)]
        return res

    def depth_to_pts3d(self):
        # Get depths and  projection params if not provided
        focals = self.get_focals()
        pp = self.get_principal_points()
        im_poses = self.get_im_poses()
        depth = self.get_depthmaps(raw=True)

        # get pointmaps in camera frame
        rel_ptmaps = _fast_depthmap_to_pts3d(depth, self._grid, focals, pp=pp)
        # project to world frame
        return geotrf(im_poses, rel_ptmaps)

    def get_pts3d(self, raw=False):
        res = self.depth_to_pts3d()
        if not raw:
            res = [dm[:h*w].view(h, w, 3) for dm, (h, w) in zip(res, self.imshapes)]
        return res
    
    def get_smplx_params(self):
        # return the parameters as dictionary
        return {
            'global_rotvec': self.human_global_rotvec.detach(),
            'relative_rotvec': self.human_relative_rotvec.detach(),
            'transl': self.human_transl.detach(),
            'shape': self.human_shape.detach(),
            'expression': self.human_expression.detach()
        }
    
    def get_smplx_output(self):
        pose = torch.cat((self.human_global_rotvec, self.human_relative_rotvec), dim=1) # (1, 53, 3)
        smplx_output = self.smplx_layer(transl=self.human_transl,
                                 pose=pose,
                                 shape=self.human_shape,
                                 K=torch.zeros((len(pose), 3, 3), device=self.device),  # dummy
                                 expression=self.human_expression,
                                 loc=None,
                                 dist=None)

        return smplx_output
    
    def save_2d_joints(self, output_dir=''):
        smplx_output = self.get_smplx_output()
        smplx_j3d = smplx_output['j3d'][:, :get_smplx_joint_names().index('jaw')] # (1, J, 3)
        smplx_j3d = smplx_j3d.repeat(self.n_imgs, 1, 1) # (self.n_imgs, J, 3)

        cam2world_4by4 = self.get_im_poses() # (self.n_imgs, 4, 4)
        world2cam_4by4 = torch.inverse(cam2world_4by4) # (self.n_imgs, 4, 4)
        K_all = self.get_intrinsics() # (self.n_imgs, 3, 3)

        proj = geotrf(world2cam_4by4, smplx_j3d)
        smplx_j2d = geotrf(K_all, proj, norm=1, ncol=2) # (self.n_imgs, J, 2)

        # draw the 2d joints on the image
        for img_idx in range(self.n_imgs):
            img = self.imgs[img_idx].copy() * 255.
            img = img.astype(np.uint8) 
            for joint in smplx_j2d[img_idx]:
                img = cv2.circle(img, (int(joint[0]), int(joint[1])), 3, (0, 255, 0), -1) 

            tmp_output_path = osp.join(output_dir, f'{img_idx}.png')
            cv2.imwrite(tmp_output_path, img[...,::-1])

        return smplx_j2d

    def forward(self):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj_pts3d = self.get_pts3d(raw=True)

        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)

        # compute the less
        li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j

        loss = li + lj

        # Hongsuk added
        if self.has_human_cue:
            cam2world_4by4 = self.get_im_poses() # (self.n_imgs, 4, 4)
            world2cam_4by4 = torch.inverse(cam2world_4by4) # (self.n_imgs, 4, 4)
            K_all = self.get_intrinsics() # (self.n_imgs, 3, 3)

            # decode the smpl mesh and joints
            smplx_output = self.get_smplx_output()
            smplx_j3d = smplx_output['j3d'][:, :get_smplx_joint_names().index('jaw')] # (1, J, 3), joints in the world coordinate from the world mesh decoded by the optimizing parameters
            # tile the smplx_j3d to the number of images
            smplx_j3d = smplx_j3d.repeat(self.n_imgs, 1, 1)

            # transform and proejct the 3d joints to the image plane 
            # let's use their code!
            proj = geotrf(world2cam_4by4, smplx_j3d)
            smplx_j2d = geotrf(K_all, proj, norm=1, ncol=2) # (self.n_imgs, J, 2)

            # compute the distance between the projected 2d joints and the given 2d joints
            # using the weight from the human_det_score and inverse of the human_bbox's area
            human_loss = ((smplx_j2d - self.human_j2d).abs().sum(dim=[1,2]) * self.human_weight).sum() # self.dist(smplx_j2d, self.human_j2d, weight=human_weight).sum() 
            human_loss = self.human_loss_weight * human_loss / self.max_area

            loss += human_loss
            
            return loss, float(human_loss)
        else:
            return loss, None

    # Hongsuk added
    def dust3r_loss(self):
        pw_poses = self.get_pw_poses()  # cam-to-world
        pw_adapt = self.get_adaptors().unsqueeze(1)
        proj_pts3d = self.get_pts3d(raw=True)

        # rotate pairwise prediction according to pw_poses
        aligned_pred_i = geotrf(pw_poses, pw_adapt * self._stacked_pred_i)
        aligned_pred_j = geotrf(pw_poses, pw_adapt * self._stacked_pred_j)

        # compute the less
        li = self.dist(proj_pts3d[self._ei], aligned_pred_i, weight=self._weight_i).sum() / self.total_area_i
        lj = self.dist(proj_pts3d[self._ej], aligned_pred_j, weight=self._weight_j).sum() / self.total_area_j

        loss = li + lj
        
        return loss

    # Hongsuk added
    def init_from_known_params_hongsuk(self, im_focals=None, im_poses=None, pts3d=None, niter_PnP=10, min_conf_thr=3):
        # set the D, P, K parameters from the known parameters (depthmaps, extrinsics, intrinsics)
        for i in range(self.n_imgs):
            cam2world = im_poses[i]
            depth = geotrf(inv(cam2world), pts3d[i])[..., 2]
            self._set_depthmap(i, depth)

            self._set_pose(self.im_poses, i, im_poses[i])
            if im_focals[i] is not None:
                self._set_focal(i, im_focals[i])
        
        # set the pw poses
        if pts3d is not None:
            # set the pairwise poses from the known 3D points
            # this looks more accurate than doing it from the camera poses and intrinsics
            for e, (i, j) in enumerate(self.edges):
                i_j = edge_str(i, j)
                # compute transform that goes from cam to world
                s, R, T = rigid_points_registration(self.pred_i[i_j], pts3d[i], conf=self.conf_i[i_j])
                self._set_pose(self.pw_poses, e, R, T, scale=s)
                # print(e, self.pw_poses[e])
        else:
            # set the pairwise poses from the known camera poses and intrinsics 
            for e, (i, j) in enumerate(tqdm(self.edges, disable=not self.verbose)):
                i_j = edge_str(i, j)

                # find relative pose for this pair
                P1 = torch.eye(4, device=device)
                msk = self.conf_i[i_j] > min(min_conf_thr, self.conf_i[i_j].min() - 0.1)
                _, P2 = fast_pnp(self.pred_j[i_j], float(im_focals[i].mean()),
                                pp=im_pp[i], msk=msk, device=device, niter_PnP=niter_PnP)

                # align the two predicted camera with the two gt cameras
                s, R, T = align_multiple_poses(torch.stack((P1, P2)), known_poses[[i, j]])
                # normally we have known_poses[i] ~= sRT_to_4x4(s,R,T,device) @ P1
                # and geotrf(sRT_to_4x4(1,R,T,device), s*P2[:3,3])
                self._set_pose(self.pw_poses, e, R, T, scale=s)
       
        print(' init loss =', float(self()[0]))

    def init_default_mst(self, niter_PnP=10, min_conf_thr=3):
        init_minimum_spanning_tree(self, niter_PnP=niter_PnP)

def _fast_depthmap_to_pts3d(depth, pixel_grid, focal, pp):
    pp = pp.unsqueeze(1)
    focal = focal.unsqueeze(1)
    assert focal.shape == (len(depth), 1, 1)
    assert pp.shape == (len(depth), 1, 2)
    assert pixel_grid.shape == depth.shape + (2,)
    depth = depth.unsqueeze(-1)
    return torch.cat((depth * (pixel_grid - pp) / focal, depth), dim=-1)


def ParameterStack(params, keys=None, is_param=None, fill=0):
    if keys is not None:
        params = [params[k] for k in keys]

    if fill > 0:
        params = [_ravel_hw(p, fill) for p in params]

    requires_grad = params[0].requires_grad
    assert all(p.requires_grad == requires_grad for p in params)

    params = torch.stack(list(params)).float().detach()
    if is_param or requires_grad:
        params = nn.Parameter(params)
        params.requires_grad_(requires_grad)
    return params


def _ravel_hw(tensor, fill=0):
    # ravel H,W
    tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

    if len(tensor) < fill:
        tensor = torch.cat((tensor, tensor.new_zeros((fill - len(tensor),)+tensor.shape[1:])))
    return tensor


def acceptable_focal_range(H, W, minf=0.5, maxf=3.5):
    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    return minf*focal_base, maxf*focal_base


def apply_mask(img, msk):
    img = img.copy()
    img[msk] = 0
    return img
