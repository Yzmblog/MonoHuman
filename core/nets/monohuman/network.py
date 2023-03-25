from cgi import print_directory
from dis import dis
from distutils.command.config import config
from distutils.log import debug
import imp
from ntpath import join
from os import PRIO_PGRP
from pickle import NONE, TRUE
from pyexpat.errors import XML_ERROR_JUNK_AFTER_DOC_ELEMENT
from threading import local
from tkinter import N
from tkinter.messagebox import NO
from turtle import back, backward, pos
from xmlrpc.client import TRANSPORT_ERROR
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import MotionBasisComputer
from core.nets.human_nerf.component_factory import \
    load_feature_extractor, \
    load_positional_embedder, \
    load_canonical_mlp, \
    load_mweight_vol_decoder, \
    load_pose_decoder, \
    load_non_rigid_motion_mlp, \
    load_blend_net, \
    load_projector
import time
from configs import cfg

import math
import numpy as np
from numpy import *

SMPL_PARENT = torch.tensor([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21])

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def _dot(a, b):
    return (a * b).sum(dim=-1, keepdim=True)

def closest_distance_to_points(
    joints, parent, points, min_index=None, individual: bool = False
):
    """Cartesian distance from points to bones (line segments).

    https://zalo.github.io/blog/closest-point-between-segments/

    :params joints: [n_joints,3]
    :params parent: SMPL_PARENT_PARAM
    :params points: If individual is true, the shape should be [..., n_bones, 3].
        else, the shape is [N, 3]
    :returns distances [..., n_bones]
    """

    if min_index is not None:
        p_index = parent[min_index]
        ret = torch.tensor(points - joints[p_index]).reshape(points.shape[0], 3)
        ret = ret.to(torch.float32)
        return ret

    heads = joints[1:]
    tails_index = parent
    tails = joints[tails_index]

    points = points.reshape(-1, 3)
    if individual:
        # the points and bones have one-to-one correspondence.
        # [..., n_bones, 3] <-> [n_bones,]
        assert points.shape[-2:] == (tails.shape[0], 3)
    else:
        points = points[..., None, :]  # [..., 1, 3]

    t = _dot(points - heads, tails - heads) / _dot(tails - heads, tails - heads)
    p = heads + (tails - heads) * torch.clamp(t, 0, 1)
    dists = torch.norm(p - points, dim=-1)
    min_index = dists.argmin(1)#N
    p_index = parent[min_index]

    points = points.squeeze(1)
    ret = torch.tensor(points - joints[p_index]).reshape(points.shape[0], 3)
    ret = ret.to(torch.float32)
    return ret, min_index

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(
                                        total_bones=cfg.total_bones)

        # motion weight volume
        self.mweight_vol_decoder = load_mweight_vol_decoder(cfg.mweight_volume.module)(
            embedding_size=cfg.mweight_volume.embedding_size,
            volume_size=cfg.mweight_volume.volume_size,
            total_bones=cfg.total_bones
        )

        self.fg_thread = cfg.fg_thread
        self.ibr_net = load_blend_net(cfg.ibrnet.module)(cfg=cfg.blend_net, n_samples=cfg.N_samples)
        self.projector = load_projector(cfg.projector.module)()
        # non-rigid motion st positional encoding
        self.get_non_rigid_embedder = \
            load_positional_embedder(cfg.non_rigid_embedder.module)

        # non-rigid motion MLP
        _, non_rigid_pos_embed_size = \
            self.get_non_rigid_embedder(cfg.non_rigid_motion_mlp.multires, 
                                        cfg.non_rigid_motion_mlp.i_embed)
        self.forward_mlp = \
            load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(
                pos_embed_size=non_rigid_pos_embed_size,
                condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
                mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
                mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
                skips=cfg.non_rigid_motion_mlp.skips,
                backward=False)
        self.backward_mlp = \
            load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(
                pos_embed_size=non_rigid_pos_embed_size,
                condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
                mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
                mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
                skips=cfg.non_rigid_motion_mlp.skips,
                backward=True)

        self.feature_extractor = load_feature_extractor(cfg.feature_extractor.module)(
            coarse_only=True
        )

        # canonical positional encoding
        get_embedder = load_positional_embedder(cfg.embedder.module)
        cnl_pos_embed_fn, cnl_pos_embed_size = \
            get_embedder(cfg.canonical_mlp.multires, 
                         cfg.canonical_mlp.i_embed)


        self.pos_embed_fn = cnl_pos_embed_fn

        self.cnl_pos_embed_size = cnl_pos_embed_size

        cnl_input_ch = cnl_pos_embed_size * 2
        skips = [4]
        self.cnl_mlp = \
            load_canonical_mlp(cfg.canonical_mlp.module)(
                input_ch=cnl_input_ch,
                output_ch = cfg.rgb_in_dim + 1, 
                mlp_depth=cfg.canonical_mlp.mlp_depth, 
                mlp_width=cfg.canonical_mlp.mlp_width,
                skips=skips)

        input_ch = cfg.rgb_in_dim * 2
        self.rgb_mlp = \
            load_canonical_mlp(cfg.canonical_mlp.module)(
                input_ch=input_ch,
                output_ch=3,
                mlp_depth=cfg.canonical_mlp.mlp_depth // 2, 
                mlp_width=cfg.canonical_mlp.mlp_width,
                skips=[2])

        # pose decoder MLP
        self.pose_decoder = \
            load_pose_decoder(cfg.pose_decoder.module)(
                embedding_size=cfg.pose_decoder.embedding_size,
                mlp_width=cfg.pose_decoder.mlp_width,
                mlp_depth=cfg.pose_decoder.mlp_depth)
    

    def deploy_mlps_to_secondary_gpus(self):
        self.cnl_mlp = self.cnl_mlp.to(cfg.secondary_gpus[0])

        if self.non_rigid_mlp:
            self.non_rigid_mlp = self.non_rigid_mlp.to(cfg.secondary_gpus[0])

        return self


    def _query_mlp(
            self,
            pos_xyz,
            pos_embed_fn,
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input, 
            T_joints,
            O_pts=None,
            O_joints=None,
            rgb_feat=None):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3)
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])
        if rgb_feat is not None:
            rgb_feat = torch.reshape(rgb_feat, [-1, rgb_feat.shape[-1]])

        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)
        
        result = self._apply_mlp_kernals(
                        pos_flat=pos_flat,
                        pos_embed_fn=pos_embed_fn,
                        non_rigid_mlp_input=non_rigid_mlp_input,
                        non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                        chunk=chunk, 
                        T_joints=T_joints,
                        input_dis=cfg.input_dis,
                        O_pts=O_pts,
                        O_joints=O_joints,
                        rgb_feat=rgb_feat)

        output = {}

        raws_flat = result['raws']
        output['raws'] = torch.reshape(
                            raws_flat, 
                            list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])

        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))


    def _apply_mlp_kernals(
            self, 
            pos_flat,
            pos_embed_fn,
            chunk,
            T_joints=None,
            input_dis=False,
            rgb_feat=None):
        raws = []

        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            xyz = pos_flat[start:end]

            import time
            if T_joints is not None and input_dis:
                dis, min_index = closest_distance_to_points(T_joints, SMPL_PARENT, xyz, rel=cfg.rel)
                rel_dis = dis
                rel_dis = rel_dis.reshape(-1, 3)
                dis_embedded = pos_embed_fn(rel_dis)
                dis_embedded = dis_embedded.reshape(xyz.shape[0], -1)
            xyz_embedded = pos_embed_fn(xyz)
            xyz_embedded = torch.cat((xyz_embedded, dis_embedded), dim=-1)

            out1 = self.cnl_mlp(
                        pos_embed=xyz_embedded)


            rgb_latent = out1[:, :-1]

            # print(rgb_feat)
            if rgb_feat is not None:
                xyz_rgb_feat = rgb_feat[start:end]

            else:
                xyz_rgb_feat = torch.zeros_like(rgb_latent).to(rgb_latent)

            rgb_input = torch.cat([rgb_latent, xyz_rgb_feat], dim=-1)


            rgb = self.rgb_mlp(rgb_input)
            density = out1[:, -1:]
            raws += [
                torch.cat([rgb, density], dim=-1)
            ]
        output = {}
        output['raws'] = torch.cat(raws, dim=0)
        return output


    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], cfg.chunk):
            ret = self._render_rays(rays_flat[i:i+cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k: torch.cat(all_ret[k], 0) for k in all_ret}

        if all_ret.__contains__('loss_consis'):
            loss_consis = torch.mean(all_ret['loss_consis'])
            if torch.isnan(loss_consis):
                all_ret.__delitem__('loss_consis')
            else:
                all_ret['loss_consis'] = loss_consis
        return all_ret


    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, bgcolor=None):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)

        dists = z_vals[...,1:] - z_vals[...,:-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)

        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        alpha = _raw2alpha(raw[...,3], dists)  # [N_rays, N_samples]
        alpha = alpha * raw_mask[:, :, 0]
        
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 
                       1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]
        

        depth_map = torch.sum(weights * z_vals, -1)
        acc_map = torch.sum(weights, -1)
        rgb_map = rgb_map + (1.-acc_map[...,None]) * bgcolor[None, :]/255.
        return rgb_map, acc_map, weights, depth_map

    @staticmethod
    def _sample_motion_fields(
            pts,
            motion_weights_vol,
            cnl_bbox_min_xyz, cnl_bbox_scale_xyz,
            output_list,
            motion_scale_Rs=None,
            motion_Ts=None,
            backward=True):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] 
        
        weights_list = []
        for i in range(motion_weights.size(0)):
            ###from observation space to canonical space
            '''
            @pts: observation space points
            @motion_weight: canonical space weight
            '''
            if backward:
                pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            else:
                pos = pts
            pos = (pos - cnl_bbox_min_xyz[None, :]) \
                            * cnl_bbox_scale_xyz[None, :] - 1.0

            weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :], 
                                    grid=pos[None, None, None, :, :],           
                                    padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None] 
            weights_list.append(weights) 
        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]
        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, 
                                                dim=-1, keepdim=True)
        weighted_motion_fields = []
        for i in range(total_bases):
            if backward:
                pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :]
            else:
                pos = torch.matmul(torch.inverse(motion_scale_Rs[i, :, :]), (pts - motion_Ts[i, :]).T).T 
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(
                        torch.stack(weighted_motion_fields, dim=0), dim=0
                        ) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum

        x_skel = x_skel.reshape(orig_shape[:1]+[3])
        backwarp_motion_weights = \
            backwarp_motion_weights.reshape(orig_shape[:1]+[total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:1]+[1])

        results = {}
        
        if 'x_skel' in output_list: # [N_rays x N_samples, 3]
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list: # [N_rays x N_samples, 1]
            results['fg_likelihood_mask'] = fg_likelihood_mask
        
        return results


    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:,3:6] 
        bounds = torch.reshape(ray_batch[...,6:8], [-1,1,2]) 
        near, far = bounds[...,0], bounds[...,1] 
        return rays_o, rays_d, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
        upper = torch.cat([mids, z_vals[...,-1:]], -1)
        lower = torch.cat([z_vals[...,:1], mids], -1)
        
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand

        return z_vals


    def _render_rays(
            self, 
            ray_batch, 
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz,
            pos_embed_fn,
            non_rigid_pos_embed_fn,
            in_K,
            in_E,
            E,
            in_motion_scale_Rs,
            in_motion_Ts,
            projector,
            src_imgs,
            featmaps,
            non_rigid_mlp_input=None,
            bgcolor=None,
            joints=None,
            canonical_joints=None,
            iter_val=None,
            dst_posevec=None,
            **_):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far)
        # print('perturb----', cfg.perturb, flush=True)
        if cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)
        #print(z_vals)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
        ori_shape = pts.shape
        pts = pts.reshape(-1, 3)
        cnl_pts, pts_mask = self.backward_deform(x_o=pts,
                            dst_posevec=dst_posevec,
                            motion_scale_Rs=motion_scale_Rs, 
                            motion_Ts=motion_Ts, 
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            non_rigid_pos_embed_fn=non_rigid_pos_embed_fn)

        fg_index = pts_mask > self.fg_thread
        
        fg_index = fg_index.squeeze(1)
        pts = pts.reshape(-1, 3)
        pts_c = cnl_pts[fg_index]
        pts_o = self.forward_deform(x_c=pts_c,
                        dst_posevec=torch.zeros_like(dst_posevec),
                        motion_scale_Rs=motion_scale_Rs, 
                        motion_Ts=motion_Ts, 
                        motion_weights_vol=motion_weights_vol,
                        cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                        cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                        non_rigid_pos_embed_fn=non_rigid_pos_embed_fn)
        distance = torch.norm(pts[fg_index] -  pts_o, dim=1)

        loss_consis = distance[distance > 0.05]
        if iter_val >= cfg.kick_in_feature:

            pts_ref = []
            for motion_scale_Rs_, motion_Ts_ in zip(in_motion_scale_Rs, in_motion_Ts):
                #pts_corr = torch.zeros_like(cnl_pts).to(cnl_pts)
                pts_src_corr = self.forward_deform(x_c=cnl_pts,
                                dst_posevec=torch.zeros_like(dst_posevec),
                                motion_scale_Rs=motion_scale_Rs_, 
                                motion_Ts=motion_Ts_, 
                                motion_weights_vol=motion_weights_vol,
                                cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                                cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                                non_rigid_pos_embed_fn=non_rigid_pos_embed_fn)
                pts_ref.append(pts_src_corr[None])
            pts_ref = torch.cat(pts_ref, dim=0)
            pts_ref = pts_ref.reshape([len(pts_ref)] + list(ori_shape))
            pts = pts.reshape(ori_shape)

            rgb_feat, mask, _ = projector.compute(pts_ref, pts, cnl_pts, in_K, in_E, E, src_imgs, featmaps)
            rgb_latant = self.ibr_net(rgb_feat, mask)

        else:
            rgb_latant = None

        cnl_pts = cnl_pts.reshape(list(ori_shape))
        pts_mask = pts_mask.reshape(list(ori_shape[:2]) + list([pts_mask.shape[-1]]))

        query_result = self._query_mlp(
                                pos_xyz=cnl_pts,
                                rgb_feat=rgb_latant,
                                non_rigid_mlp_input=non_rigid_mlp_input,
                                pos_embed_fn=pos_embed_fn,
                                non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
                                T_joints=canonical_joints,
                                O_pts=pts.reshape(-1, 3),
                                O_joints=joints)
        raw = query_result['raws']

        rgb_map, acc_map, _, depth_map = \
            self._raw2outputs(raw, pts_mask, z_vals, rays_d, bgcolor)

        return {'rgb' : rgb_map,  
                'alpha' : acc_map, 
                'depth': depth_map,
                'loss_consis':loss_consis

            }


    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(
                                        dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts


    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3),
                            correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)
    def forward(self,
                rays, 
                dst_Rs, dst_Ts, cnl_gtfms,
                in_dst_Rs, in_dst_Ts,
                motion_weights_priors,
                src_imgs,
                in_K, 
                in_E,
                K, 
                E,
                dst_posevec=None,
                near=None, far=None,
                iter_val=1e7,
                **kwargs):
       # torch.autograd.set_detect_anomaly(True)
        dst_Rs=dst_Rs[None, ...]
        dst_Ts=dst_Ts[None, ...]
        dst_posevec=dst_posevec[None, ...]
        cnl_gtfms=cnl_gtfms[None, ...]
        motion_weights_priors=motion_weights_priors[None, ...]

        non_rigid_pos_embed_fn, _ = \
            self.get_non_rigid_embedder(
                multires=cfg.non_rigid_motion_mlp.multires,                         
                is_identity=cfg.non_rigid_motion_mlp.i_embed,
                iter_val=iter_val,)

        if iter_val >= cfg.kick_in_feature:
            featmaps, _ = self.feature_extractor(src_imgs.permute(0, 3, 1, 2))
        else:
            featmaps = None

        kwargs.update({
            "pos_embed_fn": self.pos_embed_fn,
            "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "dst_posevec": dst_posevec,
            "iter_val":iter_val,
            "src_imgs":src_imgs,
            "featmaps": featmaps,
            'projector':self.projector,
            'in_K': in_K, 
            'in_E': in_E,
            'K':K, 
            'E':E,
        })

        r"""Compute motion bases between the target pose and canonical pose."""
        motion_scale_Rs, motion_Ts = self._get_motion_base(
                                            dst_Rs=dst_Rs, 
                                            dst_Ts=dst_Ts, 
                                            cnl_gtfms=cnl_gtfms)
        in_motion_scale_Rs = []
        in_motion_Ts = []
        for dst_Rs_near, dst_Ts_near in zip(in_dst_Rs, in_dst_Ts):
            motion_scale_Rs_near, motion_Ts_near = self._get_motion_base(
                                                dst_Rs=dst_Rs_near[None, ...],
                                                dst_Ts=dst_Ts_near[None, ...],
                                                cnl_gtfms=cnl_gtfms
            )
            in_motion_scale_Rs.append(motion_scale_Rs_near)
            in_motion_Ts.append(motion_Ts_near)

        motion_weights_vol = self.mweight_vol_decoder(
            motion_weights_priors=motion_weights_priors)
        motion_weights_vol=motion_weights_vol[0] # remove batch dimension

        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'in_motion_scale_Rs': in_motion_scale_Rs,
            'in_motion_Ts': in_motion_Ts,
            'motion_weights_vol': motion_weights_vol
        })

        rays_o, rays_d = rays

        rays_o = torch.reshape(rays_o, [-1,3]).float()
        rays_d = torch.reshape(rays_d, [-1,3]).float()

        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)
        
        return all_ret

    def backward_deform(self, x_o, dst_posevec, motion_scale_Rs, motion_Ts, motion_weights_vol, cnl_bbox_min_xyz, cnl_bbox_scale_xyz, non_rigid_pos_embed_fn):
        '''
        Use sample weight and non-rigid mlp to deform points from observation space to canonical space.
        params: x_o, points in canonical space
        params: dst_posevec, pose parameters
        params: motion_scale_Rs, skel transfrom matrix from observation space to canonical space. Size: 24, 3, 3
        params: motion_Ts, translation vectors from observation space to canonical space. Size: 24, 3
        params: cnl_bbox_min_xyz, minxyz coords in canonical space.
        params: cnl_bbox_scale_xyz, scale factor for xyz coords in canonical space.
        '''

        mv_output = self._sample_motion_fields(
                            pts=x_o,
                            motion_scale_Rs=motion_scale_Rs[0],
                            motion_Ts=motion_Ts[0],
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz,
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            output_list=['x_skel', 'fg_likelihood_mask'],
                            backward=True)

        pts_mask = mv_output['fg_likelihood_mask']
        x_c = mv_output['x_skel']

        x_c = x_c.reshape(-1, 3)
        pts_mask = pts_mask.reshape(-1, 1)

        non_rigid_embed_xyz = non_rigid_pos_embed_fn(x_c)
        result = self.backward_mlp(
            pos_embed=non_rigid_embed_xyz,
            pos_xyz=x_c,
            condition_code=self._expand_input(dst_posevec, x_c.shape[0])
        )
        x_c = result['xyz']

        return x_c, pts_mask


    def forward_deform(self, x_c, dst_posevec, motion_scale_Rs, motion_Ts, motion_weights_vol, cnl_bbox_min_xyz, cnl_bbox_scale_xyz, non_rigid_pos_embed_fn):
        '''
        Inverse of backward deformation to deform points from canonical space to observation space.
        '''
        mv_output = self._sample_motion_fields(
                            pts=x_c,
                            motion_scale_Rs=motion_scale_Rs[0], 
                            motion_Ts=motion_Ts[0], 
                            motion_weights_vol=motion_weights_vol,
                            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
                            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
                            output_list=['x_skel'],
                            backward=False)

        x_o = mv_output['x_skel']
        x_o = x_o.reshape(-1, 3)


        non_rigid_embed_xyz = non_rigid_pos_embed_fn(x_o)
        result = self.forward_mlp(
            pos_embed=non_rigid_embed_xyz,
            pos_xyz=x_o,
            condition_code=self._expand_input(dst_posevec, x_o.shape[0])
        )
        x_o = result['xyz']

        return x_o

