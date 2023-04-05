import os
import pickle

import torch.nn.functional as F

import numpy as np
import cv2
import torch
import torch.utils.data

from core.utils.image_util import load_image
from core.utils.body_util import \
    body_pose_to_body_RTs, \
    get_canonical_global_tfms, \
    approx_gaussian_bone_volumes
from core.utils.file_util import list_files, split_path
from core.utils.camera_util import \
    apply_global_tfm_to_camera, \
    rotate_camera_by_frame_idx, \
    get_rays_from_KRT, \
    rays_intersect_3d_bbox

from configs import cfg
from third_parties.smpl.smpl_numpy import SMPL

MODEL_DIR = 'third_parties/smpl/models'

class Dataset(torch.utils.data.Dataset):

    ROT_CAM_PARAMS = {
        'zju_mocap': {'rotate_axis': 'z', 'inv_angle': True},
        'wild': {'rotate_axis': 'y', 'inv_angle': False}
    }


    def __init__(
            self, 
            dataset_path,
            index_a,
            index_b,
            keyfilter=None,
            maxframes=-1,
            bgcolor=None,
            ray_shoot_mode='image',
            skip=1,
            **_):
        print('[Dataset Path]', dataset_path) 

        self.src_type = 'zju_mocap'

        self.smpl_model = SMPL(sex='neutral', model_dir=MODEL_DIR)

        self.dataset_path = dataset_path
        self.image_dir = os.path.join(dataset_path, 'images')

        self.total_frames = cfg.render_frames

        self.canonical_joints, self.canonical_bbox = \
            self.load_canonical_joints()

        if 'motion_weights_priors' in keyfilter:
            self.motion_weights_priors = \
                approx_gaussian_bone_volumes(
                    self.canonical_joints,   
                    self.canonical_bbox['min_xyz'],
                    self.canonical_bbox['max_xyz'],
                    grid_size=cfg.mweight_volume.volume_size).astype('float32')


        self.cameras = self.load_train_cameras()
        self.mesh_infos = self.load_train_mesh_infos()

        framelist = self.load_train_frames()

        ### just for quick
        self.train_frame_idx = cfg.freeview.frame_idx

        self.framelist = framelist[::skip]
        self.train_frame_name = self.framelist[self.train_frame_idx]

        if maxframes > 0:
            self.framelist = self.framelist[:maxframes]

        bgcolor = None
        self.bgcolor = bgcolor if bgcolor is not None else [0., 0., 0.]
        self.keyfilter = keyfilter
        self.ray_shoot_mode = ray_shoot_mode
        self.index_a = index_a
        self.index_b = index_b

        self.train_camera = self.cameras[framelist[self.train_frame_idx]]


    def load_canonical_joints(self):
        cl_joint_path = os.path.join(self.dataset_path, 'canonical_joints.pkl')
        with open(cl_joint_path, 'rb') as f:
            cl_joint_data = pickle.load(f)
        canonical_joints = cl_joint_data['joints'].astype('float32')
        canonical_bbox = self.skeleton_to_bbox(canonical_joints)

        return canonical_joints, canonical_bbox

    def load_train_cameras(self):
        cameras = None
        with open(os.path.join(self.dataset_path, 'cameras.pkl'), 'rb') as f: 
            cameras = pickle.load(f)
        return cameras

    @staticmethod
    def skeleton_to_bbox(skeleton):
        min_xyz = np.min(skeleton, axis=0) - cfg.bbox_offset
        max_xyz = np.max(skeleton, axis=0) + cfg.bbox_offset

        return {
            'min_xyz': min_xyz,
            'max_xyz': max_xyz
        }

    def load_train_mesh_infos(self):
        mesh_infos = None
        with open(os.path.join(self.dataset_path, 'mesh_infos.pkl'), 'rb') as f:   
            mesh_infos = pickle.load(f)

        for frame_name in mesh_infos.keys():
            bbox = self.skeleton_to_bbox(mesh_infos[frame_name]['joints'])
            mesh_infos[frame_name]['bbox'] = bbox
            
        return mesh_infos

    def load_train_frames(self):
        img_paths = list_files(os.path.join(self.dataset_path, 'images'),
                               exts=['.png'])
        return [split_path(ipath)[1] for ipath in img_paths]
    
    def query_dst_skeleton(self, frame_name):
        return {
            'poses': self.mesh_infos[frame_name]['poses'].astype('float32'),
            'dst_tpose_joints': \
                self.mesh_infos[frame_name]['tpose_joints'].astype('float32'),
            'bbox': self.mesh_infos[frame_name]['bbox'].copy(),
            'Rh': self.mesh_infos[frame_name]['Rh'].astype('float32'),
            'Th': self.mesh_infos[frame_name]['Th'].astype('float32')
        }

    @staticmethod
    def select_rays(select_inds, rays_o, rays_d, ray_img, near, far):
        rays_o = rays_o[select_inds]
        rays_d = rays_d[select_inds]
        ray_img = ray_img[select_inds]
        near = near[select_inds]
        far = far[select_inds]
        return rays_o, rays_d, ray_img, near, far
    
    
    def load_image(self, frame_name, bg_color):
        imagepath = os.path.join(self.image_dir, '{}.png'.format(frame_name))
        orig_img = np.array(load_image(imagepath))

        maskpath = os.path.join(self.dataset_path, 
                                'masks', 
                                '{}.png'.format(frame_name))

        alpha_mask = np.array(load_image(maskpath))


        # undistort image
        if frame_name in self.cameras and 'distortions' in self.cameras[frame_name]:
            K = self.cameras[frame_name]['intrinsics']
            D = self.cameras[frame_name]['distortions']
            orig_img = cv2.undistort(orig_img, K, D)
            alpha_mask = cv2.undistort(alpha_mask, K, D)

        alpha_mask = alpha_mask / 255.
        img = alpha_mask * orig_img + (1.0 - alpha_mask) * bg_color[None, None, :]
        if cfg.resize_img_scale != 1.:
            img = cv2.resize(img, None, 
                                fx=cfg.resize_img_scale,
                                fy=cfg.resize_img_scale,
                                interpolation=cv2.INTER_LANCZOS4)
            alpha_mask = cv2.resize(alpha_mask, None, 
                                    fx=cfg.resize_img_scale,
                                    fy=cfg.resize_img_scale,
                                    interpolation=cv2.INTER_LINEAR)
                                
        return img, alpha_mask


    def get_total_frames(self):
        return len(self.framelist)


    def __len__(self):

        return self.total_frames


    def get_freeview_camera(self, frame_idx, total_frames, trans=None):
        E = rotate_camera_by_frame_idx(
                extrinsics=self.train_camera['extrinsics'], 
                frame_idx=frame_idx,
                period=total_frames,
                trans=trans,
                **self.ROT_CAM_PARAMS[self.src_type])
        K = self.train_camera['intrinsics'].copy()
        K[:2] *= cfg.resize_img_scale
        return K, E

    def __getitem__(self, idx):

        frame_name = self.train_frame_name
        results = {
            'frame_name': frame_name
        }
        
        frame_name_a = f'frame_{self.index_a:06d}'
        frame_name_b = f'frame_{self.index_b:06d}'
        
        in_K = []
        in_E = []
        in_dst_poses = []
        in_dst_tposes_joints = []

        in_frame_name = [frame_name_a, frame_name_b]
        in_index = [self.index_a, self.index_b]

        if self.bgcolor is None:
            bgcolor = (np.zeros(3) * 255.).astype('float32')

        else:
            bgcolor = np.array(self.bgcolor, dtype='float32')


        img, _ = self.load_image(frame_name, bgcolor)
        img = (img / 255.).astype('float32')
        
        
        img_a, _ = self.load_image(frame_name_a, np.array([0., 0., 0.]))
        img_a = (img_a / 255.).astype('float32')
        img_b, _ = self.load_image(frame_name_b, np.array([0., 0., 0.]))
        img_b = (img_b / 255.).astype('float32')
        src_img = np.array([img_a, img_b])
        
        img = np.zeros_like(img_a)

        poses = self.mesh_infos[frame_name]['poses']
        betas = self.mesh_infos[frame_name]['beats']
        _, _, joints = self.smpl_model(poses, betas)

        H, W = img.shape[0:2]

        dst_skel_info = self.query_dst_skeleton(frame_name)
        dst_tpose_joints = dst_skel_info['dst_tpose_joints']

        dst_bbox = self.skeleton_to_bbox(self.mesh_infos[frame_name]['joints'])
        dst_poses = poses

        for in_idx, in_name in zip(in_index, in_frame_name):
            assert in_name in self.cameras
            K_ = self.cameras[in_name]['intrinsics'].copy()
            K_[:2] *= cfg.resize_img_scale

            in_skel_info = self.query_dst_skeleton(in_name)
            pose_ = in_skel_info['poses']
            tpose_joints_ = in_skel_info['dst_tpose_joints']

            E_ = self.cameras[in_name]['extrinsics']
            E_ = apply_global_tfm_to_camera(
                    E=E_, 
                    Rh=in_skel_info['Rh'],
                    Th=in_skel_info['Th'])

            in_K.append(K_.astype('float32'))
            in_E.append(E_.astype('float32'))
            in_dst_poses.append(pose_)
            in_dst_tposes_joints.append(tpose_joints_)

        K, E = self.get_freeview_camera(
                        frame_idx=idx,
                        total_frames=self.total_frames,
                        trans=dst_skel_info['Th'])

        E = apply_global_tfm_to_camera(
                E=E, 
                Rh=dst_skel_info['Rh'],
                Th=dst_skel_info['Th'])


        E = E.astype('float32')
        R = E[:3, :3]
        T = E[:3, 3]
        
        rays_o, rays_d = get_rays_from_KRT(H, W, K, R, T)
        ray_img = img.reshape(-1, 3) 
        rays_o = rays_o.reshape(-1, 3) # (H, W, 3) --> (N_rays, 3)
        rays_d = rays_d.reshape(-1, 3)

        # (selected N_samples, ), (selected N_samples, ), (N_samples, )
        near, far, ray_mask = rays_intersect_3d_bbox(dst_bbox, rays_o, rays_d)
        rays_o = rays_o[ray_mask]
        rays_d = rays_d[ray_mask]
        ray_img = ray_img[ray_mask]

        near = near[:, None].astype('float32')
        far = far[:, None].astype('float32')

        batch_rays = np.stack([rays_o, rays_d], axis=0) 

        if 'rays' in self.keyfilter:
            results.update({
                'img_width': W,
                'img_height': H,
                'ray_mask': ray_mask,
                'rays': batch_rays,
                'near': near,
                'far': far,
                'bgcolor': bgcolor, 
                'ori_img': img,
                'src_imgs':src_img,
                'joints': joints,
                'canonical_joints': self.canonical_joints
                })

        if 'target_rgbs' in self.keyfilter:
            results['target_rgbs'] = ray_img

        if 'motion_bases' in self.keyfilter:
            dst_Rs, dst_Ts = body_pose_to_body_RTs(
                    dst_poses, dst_tpose_joints
                )
            cnl_gtfms = get_canonical_global_tfms(
                            self.canonical_joints)
            results.update({
                'dst_Rs': dst_Rs,
                'dst_Ts': dst_Ts,
                'cnl_gtfms': cnl_gtfms
            })

            in_dst_Rs = []
            in_dst_Ts = []
            for i in range(len(in_dst_poses)):
                dst_Rs_, dst_Ts_ = body_pose_to_body_RTs(
                    in_dst_poses[i], in_dst_tposes_joints[i]
                )
                in_dst_Rs.append(dst_Rs_)
                in_dst_Ts.append(dst_Ts_)
            results.update(
                {
                    'in_dst_Rs': np.array(in_dst_Rs),
                    'in_dst_Ts': np.array(in_dst_Ts)
                }
            )

        if 'motion_weights_priors' in self.keyfilter:
            results['motion_weights_priors'] = self.motion_weights_priors.copy()

        # get the bounding box of canonical volume
        if 'cnl_bbox' in self.keyfilter:
            min_xyz = self.canonical_bbox['min_xyz'].astype('float32')
            max_xyz = self.canonical_bbox['max_xyz'].astype('float32')
            results.update({
                'cnl_bbox_min_xyz': min_xyz,
                'cnl_bbox_max_xyz': max_xyz,
                'cnl_bbox_scale_xyz': 2.0 / (max_xyz - min_xyz)
            })
            assert np.all(results['cnl_bbox_scale_xyz'] >= 0)

        if 'dst_posevec_69' in self.keyfilter:
            # 1. ignore global orientation
            # 2. add a small value to avoid all zeros
            dst_posevec_69 = dst_poses[3:] + 1e-2
            results.update({
                'dst_posevec': dst_posevec_69,
            })

        in_dst_posevec = []
        for posevec in in_dst_poses:
            in_dst_posevec_69 = posevec[3:] + 1e-2
            in_dst_posevec.append(in_dst_posevec_69)
        results.update({
            'in_dst_posevec': np.array(in_dst_posevec)
        })

        results.update({
            'in_K': np.array(in_K),
            'in_E': np.array(in_E),
            'E': E,
            'K': K
        }
        )

        return results


