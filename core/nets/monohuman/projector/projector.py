import torch
import torch.nn.functional as F

class Projector():
    '''
    find correspondences in correspondences bank
    '''
    def __init__(self) -> None:
        pass
    
    def compute_projections(self, pts, in_K, in_E):
        '''
        project 3d points to pixels
        :param pts: N, 3
        :param in_K: 2, 3, 3
        :param in_E: 2, 4, 4
        :return location: 2, N, 2; mask: 2, N
        '''
        original_shape = pts.shape[1:3]
        num_views = len(pts)
        pts = pts.reshape(num_views, -1, 3)
        Rt = in_E
        tvec = Rt[:, :3, 3]
        R = Rt[:, :3, :3]
        v_cam = torch.einsum(
            "...ik,...kj->...ij", pts, R.transpose(1, 2)
        )
        v_cam = v_cam + tvec[:, None, :]
        v_project = torch.einsum(
            "...ik,...kj->...ij", v_cam, in_K.transpose(1, 2)
        )
        v_pixel = v_project[:, :, :2] / torch.clamp(v_project[:, :, 2:3], min=1e-8)
        v_pixel = torch.clamp(v_pixel, min=-1e6, max=1e6)
        mask = v_project[..., 2] > 0

        return v_pixel.reshape((num_views, ) + original_shape + (2, )), \
               mask.reshape((num_views, ) + original_shape)


    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def inbound(self, pixel_locations, h, w):
        '''
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        '''
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) &\
               (pixel_locations[..., 1] >= 0)

    def compute(self,  pts, in_K, in_E, train_imgs, featmaps):
        '''
        :param pts,: [N, 3]
        :param in_K: 2, 3, 3
        :param in_E: 2, 4, 4
        :param train_imgs: [1, 2, h, w, 3]
        :param featmaps: [n_views, d, h, w]
        :return: rgb_feat_sampled: [n_rays, n_samples, 3+n_feat],train_imgs
        '''

        train_imgs = train_imgs.permute(0, 3, 1, 2)  # [n_views, 3, h, w]

        h, w = train_imgs[0].shape[-2:]
        # compute the projection of the query points to each reference image
        pixel_locations, mask_in_front = self.compute_projections(pts, in_K, in_E)
        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   # [n_views, n_rays, n_samples, 2]

        # rgb sampling
        rgbs_sampled = F.grid_sample(train_imgs, normalized_pixel_locations, align_corners=True)
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, 3]

        # deep feature sampling
        feat_sampled = F.grid_sample(featmaps, normalized_pixel_locations, align_corners=True)
        feat_sampled = feat_sampled.permute(2, 3, 0, 1)  # [n_rays, n_samples, n_views, d]
        rgb_feat_sampled = torch.cat([rgb_sampled, feat_sampled], dim=-1)   # [n_rays, n_samples, n_views, d+3]

        # mask
        inbound = self.inbound(pixel_locations, h, w)
        mask = (inbound * mask_in_front).float().permute(1, 2, 0)[..., None]   # [n_rays, n_samples, n_views, 1]
        return rgb_feat_sampled, mask, pixel_locations
