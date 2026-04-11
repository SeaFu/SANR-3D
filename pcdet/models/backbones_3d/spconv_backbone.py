import torch.nn as nn
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
import math
import torch.nn.init as init

from functools import partial

from pcdet.datasets.augmentor.X_transform import X_TRANS
from ...utils.spconv_utils import replace_feature, spconv


def layer_voxel_discard(sparse_t, rat=0.15):
    """
    Layer-level stochastic voxel discard.
    Randomly drops a fraction of active voxels during training.
    """
    if rat <= 0:
        return

    length = sparse_t.features.shape[0]
    # Randomly keep (1 - rat) voxels
    keep_num = int(length * (1 - rat))
    if keep_num <= 0:
        return
    perm = np.random.permutation(length)
    keep_inds = perm[:keep_num]
    keep_inds = torch.from_numpy(keep_inds).to(sparse_t.features.device)

    sparse_t = replace_feature(sparse_t, sparse_t.features[keep_inds])
    sparse_t.indices = sparse_t.indices[keep_inds]


def index2points(indices, pts_range=[0, -40, -3, 70.4, 40, 1], voxel_size=[0.05, 0.05, 0.05], stride=8):
    """
    Convert voxel indices to voxel-center coordinates.
    indices: [N, 4] = [batch_idx, z, y, x]
    """
    voxel_size = np.array(voxel_size) * stride
    min_x = pts_range[0] + voxel_size[0] / 2
    min_y = pts_range[1] + voxel_size[1] / 2
    min_z = pts_range[2] + voxel_size[2] / 2

    new_indices = indices.clone().float()
    indices_float = indices.clone().float()

    # Map [z, y, x] indices back to (x, y, z)
    new_indices[:, 1] = indices_float[:, 3] * voxel_size[0] + min_x
    new_indices[:, 2] = indices_float[:, 2] * voxel_size[1] + min_y
    new_indices[:, 3] = indices_float[:, 1] * voxel_size[2] + min_z
    return new_indices


def input_voxel_discard(sparse_t,
                        discard_rate=0.9,
                        num_bins=10,
                        close_distance=30.0,
                        voxel_size=[0.05, 0.05, 0.05],
                        pts_range=[0, -40, -3, 70.4, 40, 1],
                        stride=1):
    """
    Bin-based input voxel discard.
    Drops more near-range voxels and keeps more far-range voxels.
      - discard_rate: fraction of near-range voxels to discard
      - num_bins: number of distance bins
      - close_distance: threshold for near-range voxels
      - voxel_size, pts_range, stride: used to recover voxel centers
    """

    if discard_rate <= 0:
        return

    coords = sparse_t.indices  # (N, 4) = [batch_idx, z, y, x]
    device = coords.device
    # Compute voxel-center coordinates
    points = index2points(coords, pts_range=pts_range, voxel_size=voxel_size, stride=stride)
    # points[:, 1:4] = (x, y, z) in LiDAR coordinates

    # Use planar distance to the LiDAR origin
    # Adjust this if a different distance definition is preferred
    # Here we use xy-plane distance
    diff_x = points[:, 1]
    diff_y = points[:, 2]
    dist_xy = torch.sqrt(diff_x**2 + diff_y**2).cpu().numpy()

    # Split voxels into near / far by distance
    # Apply random keeping to near voxels and keep far voxels
    # This can be extended to multiple bins if needed
    keep_mask = np.ones_like(dist_xy, dtype=bool)

    # Simple version: voxels within close_distance are treated as near-range
    # Random discard is applied only to near voxels
    near_mask = dist_xy <= close_distance
    near_inds = np.where(near_mask)[0]
    n_near = near_inds.shape[0]
    if n_near > 0:
        keep_num_near = int(n_near * (1.0 - discard_rate))
        if keep_num_near <= 0:
            # Drop all near-range voxels
            keep_mask[near_inds] = False
        else:
            perm = np.random.permutation(n_near)
            keep_near_inds = near_inds[perm[:keep_num_near]]
            # Drop near voxels not selected to keep
            drop_inds = set(near_inds) - set(keep_near_inds)
            for di in drop_inds:
                keep_mask[di] = False

    final_keep_inds = np.where(keep_mask)[0]
    final_keep_inds = torch.from_numpy(final_keep_inds).long().to(device)

    # Update sparse tensor
    new_features = sparse_t.features[final_keep_inds]
    new_indices = sparse_t.indices[final_keep_inds]
    sparse_t = replace_feature(sparse_t, new_features)
    sparse_t.indices = new_indices


def index2uv(d3_indices, batch_size, calib, stride, x_trans_train, trans_param):
    """
    Project 3D voxel indices to image-plane uv coordinates.
    Uses voxel centers from index2points and downsamples by stride.
    """
    new_uv = d3_indices.new(size=(d3_indices.shape[0], 3))
    depth = d3_indices.new(size=(d3_indices.shape[0], 1)).float()
    for b_i in range(batch_size):
        cur_in = d3_indices[d3_indices[:, 0] == b_i]
        cur_pts = index2points(cur_in, stride=stride)  # (N, 4), columns 1:4 are (x, y, z)
        if trans_param is not None:
            transed = x_trans_train.backward_with_param({
                'points': cur_pts[:, 1:4],
                'transform_param': trans_param[b_i]
            })
            cur_pts = transed['points']
        else:
            cur_pts = cur_pts[:, 1:4]

        # LiDAR -> camera -> image projection
        pts_rect = calib[b_i].lidar_to_rect_cuda(cur_pts)
        pts_img, pts_rect_depth = calib[b_i].rect_to_img_cuda(pts_rect)
        pts_img = pts_img.int()

        new_uv[d3_indices[:, 0] == b_i, 1:3] = pts_img
        depth[d3_indices[:, 0] == b_i, 0] = pts_rect_depth[:]

    new_uv[:, 0] = d3_indices[:, 0]
    # Downsample image coordinates by stride
    new_uv[:, 1] = torch.clamp(new_uv[:, 1], min=0, max=1399) // stride
    new_uv[:, 2] = torch.clamp(new_uv[:, 2], min=0, max=599) // stride
    return new_uv, depth


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size,
                                 bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding, bias=False,
                                   indice_key=indice_key)
        relu = nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size,
                                          indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )
    return m


def post_act_block2d(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                     conv_type='subm', norm_fn=None):
    """
    2D sparse block used by NRConv.
    """
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size,
                                 bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size,
                                   stride=stride, padding=padding, bias=False,
                                   indice_key=indice_key)
        relu = nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size,
                                          indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )
    return m

class NRConvBlock(nn.Module):
    """
    Noise-Resistant Convolution:
    Combine 3D and 2D sparse convolutions to reduce depth noise.
    """
    def __init__(self,
                 input_c=16,
                 output_c=16,
                 stride=1,
                 padding=1,
                 indice_key='vir1',
                 conv_depth=False,
                 debug=False,                # export a 2D heatmap during forward
                 debug_out_dir='./debug_vis', # heatmap output directory
                 use_ca=False, 
                 n_head=4,
                 **kwargs
                 ):
        super(NRConvBlock, self).__init__()
        d2_dilation = 2           # fixed dilation
        d2_padding  = 2           # matching padding
        self.stride = stride
        self.debug = debug
        self.debug_out_dir = debug_out_dir
        self.conv_depth = conv_depth

        # Heatmap file index
        self.save_count = 0

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # Downsample first if stride > 1
        if self.stride > 1:
            self.down_layer = spconv.SparseSequential(
                spconv.SparseConv3d(
                    input_c, output_c, 3,
                    stride=stride, padding=padding, bias=False,
                    indice_key=('sp' + indice_key)
                ),
                norm_fn(output_c),
                nn.ReLU()
            )
            c1 = output_c
        else:
            c1 = input_c

        if self.conv_depth:
            c1 += 4
        c2 = output_c

        # 3D submanifold conv
        self.d3_conv1 = spconv.SparseSequential(
            spconv.SubMConv3d(c1, c2 // 2, 3, padding=1, bias=False, indice_key=('subm1' + indice_key)),
            norm_fn(c2 // 2),
            nn.ReLU()
        )
        self.d3_conv2 = spconv.SparseSequential(
            spconv.SubMConv3d(c2 // 2, c2 // 2, 3, padding=1, bias=False, indice_key=('subm2' + indice_key)),
            norm_fn(c2 // 2),
            nn.ReLU()
        )

        # 2D conv block
        self.d2_conv1 = spconv.SparseSequential(
            spconv.SubMConv2d(c2 // 2, c2 // 2, 3,
                                padding=d2_padding,              # =2
                                dilation=d2_dilation,            # =2
                                bias=False, indice_key=('subm3' + indice_key)),
            norm_fn(c2 // 2),
            nn.ReLU()
        )
        self.d2_conv2 = spconv.SparseSequential(
            spconv.SubMConv2d(c2 // 2, c2 // 2, 3,
                                padding=d2_padding,              # =2
                                dilation=d2_dilation,            # =2
                                bias=False, indice_key=('subm4' + indice_key)),
            norm_fn(c2 // 2),
            nn.ReLU()
        )
#-----------------------------------------# Single-layer multi-head cross-attention
        self.use_ca = use_ca
        if self.use_ca:
            self.n_head  = n_head
            self.d_model  = output_c // 2           # C
            assert self.d_model % self.n_head == 0,\
                    "d_model must be divisible by n_head"
            self.d_k      = self.d_model // self.n_head
            # Q from 3D, K/V from 2D
            self.q  = nn.Linear(self.d_model, self.d_model, bias=False)
            self.kv = nn.Linear(self.d_model, self.d_model * 2, bias=False)
            self.proj = nn.Linear(self.d_model, self.d_model, bias=False)
            for m in (self.q, self.kv, self.proj):
                init.xavier_uniform_(m.weight, gain=1e-3) 

    def forward(self, sp_tensor, batch_size, calib, stride, x_trans_train, trans_param):
        # Downsample first if stride > 1
        if self.stride > 1:
            sp_tensor = self.down_layer(sp_tensor)

        # (1) 3D conv
        d3_feat1 = self.d3_conv1(sp_tensor)
        d3_feat2 = self.d3_conv2(d3_feat1)

        # (2) Project to 2D
        uv_coords, depth = index2uv(
            d3_feat2.indices, batch_size, calib,
            stride, x_trans_train, trans_param
        )
        d2_sp_tensor = spconv.SparseConvTensor(
            features=d3_feat2.features,
            indices=uv_coords.int(),
            spatial_shape=[1600, 600],  # simplified 2D plane
            batch_size=batch_size
        )
        # (3) 2D conv
        d2_feat1 = self.d2_conv1(d2_sp_tensor)
        d2_feat2 = self.d2_conv2(d2_feat1)





        # (4) Fuse 2D features back into 3D features
        if self.use_ca:
            f3d = d3_feat2.features            # (N, C)
            f2d = d2_feat2.features            # (N, C)

            q  = self.q(f3d)                                   # (N, C)
            k, v = self.kv(f2d).chunk(2, dim=-1)               # (N, C), (N, C)

            # Reshape for multi-head attention
            N = q.size(0)
            q = q.view(N, self.n_head, self.d_k)
            k = k.view(N, self.n_head, self.d_k)
            v = v.view(N, self.n_head, self.d_k)

            attn = (q * k).sum(-1) / math.sqrt(self.d_k)       # (N, n_head)
            attn = torch.softmax(attn, dim=-1).unsqueeze(-1)   # (N, n_head, 1)

            fuse = self.proj((attn * v).contiguous()
                             .view(N, self.d_model))           # (N, C)

            fused_feat = torch.cat([f3d + fuse, f2d], dim=-1)   # (N, 2C)
        else:
            fused_feat = torch.cat([d3_feat2.features,
                                    d2_feat2.features], dim=-1)

        d3_feat3 = replace_feature(d3_feat2, fused_feat)

        #d3_feat3 = replace_feature(
        #    d3_feat2,
        #    torch.cat([d3_feat2.features, d2_feat2.features], dim=-1)
        #)






        # (5) Optionally export a 2D heatmap
        if self.debug:
            # Use batch 0 by default
            self.save_2d_heatmap(
                d2_feat2,
                batch_idx=0,
                prefix='NRConv_debug'
            )

        return d3_feat3

    def save_2d_heatmap(self, sp_tensor2d, batch_idx=0, prefix='NRConv_debug'):
        """
        sp_tensor2d: SparseConvTensor with shape (N, C2) in .features
                     and (N, 3 or 4) in .indices
        """
        feats = sp_tensor2d.features.detach().cpu().numpy()   # (N, C)
        inds  = sp_tensor2d.indices.detach().cpu().numpy()    # (N, 4) or (N,3)
        shape2d = sp_tensor2d.spatial_shape                   # [1600, 600]

        H, W = shape2d[0], shape2d[1]  # note: [1600, 600], you might transpose later

        # Keep only the selected batch
        mask = (inds[:, 0] == batch_idx)
        feats_b = feats[mask]  # (M, C)
        inds_b  = inds[mask]   # (M, 4) or (M, 3)

        # Compute heatmap values, e.g. L2 norm
        norms = np.linalg.norm(feats_b, axis=1)  # (M,)

        # Create a 2D heatmap
        heatmap = np.zeros((H, W), dtype=np.float32)

        # inds_b[:, 1], inds_b[:, 2] => x, y
        x_ = inds_b[:, 1]
        y_ = inds_b[:, 2]

        for i in range(len(x_)):
            xx, yy = x_[i], y_[i]
            if 0 <= xx < H and 0 <= yy < W:
                heatmap[xx, yy] = norms[i]

        # optional: transpose if you want (y,x) => (H, W)
        # heatmap = heatmap.T  # depends on your definition

        os.makedirs(self.debug_out_dir, exist_ok=True)

        # Use save_count to keep filenames unique
        out_png = os.path.join(
            self.debug_out_dir,
            f"{prefix}_batch{batch_idx}_heatmap_{self.save_count}.png"
        )
        plt.figure(figsize=(10,8))
        plt.imshow(heatmap, cmap='jet')
        plt.colorbar()
        plt.title(f"{prefix}: 2D feature norm (batch={batch_idx}) iter={self.save_count}")
        plt.savefig(out_png, dpi=150)
        plt.close()

        print(f"[DEBUG] Save 2D heatmap => {out_png}")

        # Increase the file index
        self.save_count += 1


class VirConvL8x(nn.Module):
    """
    VirConv-style backbone with input StVD and layer-level discard.
    Lightweight backbone variant.
    """
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.x_trans_train = X_TRANS()

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features = model_cfg.OUT_FEATURES
        self.layer_discard_rate = model_cfg.LAYER_DISCARD_RATE

        num_filters = model_cfg.NUM_FILTERS
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        # Sparse spatial shape after voxelization
        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        # Four NRConv blocks with strides (1, 2, 2, 2)
        self.vir_conv1 = NRConvBlock(input_channels, num_filters[0],
                                     stride=1, indice_key='vir1')
        self.vir_conv2 = NRConvBlock(num_filters[0], num_filters[1],
                                     stride=2, indice_key='vir2')
        self.vir_conv3 = NRConvBlock(
            num_filters[1], num_filters[2],
            stride=2, padding=(0, 1, 1),
            indice_key='vir3',
            use_ca=True,
            n_head=8
        )
        self.vir_conv4 = NRConvBlock(
            num_filters[2], num_filters[3],
            stride=2, padding=(0, 1, 1),
            indice_key='vir4',
            use_ca=True,
            n_head=4
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            spconv.SparseConv3d(num_filters[3], self.out_features,
                                (3, 1, 1), stride=(2, 1, 1),
                                padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )

        self.num_point_features = self.out_features
        if self.return_num_features_as_dict:
            num_point_features = {
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3]
            }
            self.num_point_features = num_point_features





    def forward(self, batch_dict):
        """
        batch_dict includes:
          - voxel_features / voxel_coords
          - batch_size
          - calib: camera-LiDAR calibration
          - transform_param: optional augmentation parameters
        """
        if 'transform_param' in batch_dict:
            trans_param_all = batch_dict['transform_param']
            rot_num = trans_param_all.shape[1]  # may include multiple transforms
        else:
            rot_num = 1

        batch_size = batch_dict['batch_size']

        for i in range(rot_num):
            rot_num_id = '' if i == 0 else str(i)

            newvoxel_features = batch_dict['voxel_features' + rot_num_id]
            newvoxel_coords = batch_dict['voxel_coords' + rot_num_id]

            # Optional place for feature masking or tagging
            # newvoxel_features[:, 4:7] = 0
            # newvoxel_features[:, 7] *= 100

            # Build SparseConvTensor
            newinput_sp_tensor = spconv.SparseConvTensor(
                features=newvoxel_features,
                indices=newvoxel_coords.int(),
                spatial_shape=self.sparse_shape,
                batch_size=batch_size
            )

            # Input-stage bin-based discard (StVD)
            # Example: discard_rate=0.9 drops 90% of near-range voxels
            if self.training:
                input_voxel_discard(
                    sparse_t=newinput_sp_tensor,
                    discard_rate=0.9,
                    num_bins=10,
                    close_distance=25.0,   # was 30
                    voxel_size=[0.05, 0.05, 0.05],
                    pts_range=[0, -40, -3, 70.4, 40, 1],
                    stride=1
                )

            # Optional augmentation parameters
            if 'transform_param' in batch_dict:
                trans_param = batch_dict['transform_param'][:, i, :]
            else:
                trans_param = None

            calib = batch_dict['calib']

            # Stage 1
            newx_conv1 = self.vir_conv1(newinput_sp_tensor,
                                        batch_size, calib,
                                        1, self.x_trans_train, trans_param)
            # layer-level discard
            if self.training and self.layer_discard_rate > 0:
                layer_voxel_discard(newx_conv1, self.layer_discard_rate)

            # Stage 2
            newx_conv2 = self.vir_conv2(newx_conv1,
                                        batch_size, calib,
                                        2, self.x_trans_train, trans_param)
            if self.training and self.layer_discard_rate > 0:
                layer_voxel_discard(newx_conv2, self.layer_discard_rate)

            # Stage 3
            newx_conv3 = self.vir_conv3(newx_conv2,
                                        batch_size, calib,
                                        4, self.x_trans_train, trans_param)
            if self.training and self.layer_discard_rate > 0:
                layer_voxel_discard(newx_conv3, self.layer_discard_rate)

            # Stage 4
            newx_conv4 = self.vir_conv4(newx_conv3,
                                        batch_size, calib,
                                        8, self.x_trans_train, trans_param)
            # Output convolution
            out = self.conv_out(newx_conv4)

            batch_dict.update({
                'encoded_spconv_tensor' + rot_num_id: out,
                'encoded_spconv_tensor_stride' + rot_num_id: 8,
            })
            batch_dict.update({
                'multi_scale_3d_features' + rot_num_id: {
                    'x_conv1': newx_conv1,
                    'x_conv2': newx_conv2,
                    'x_conv3': newx_conv3,
                    'x_conv4': newx_conv4,
                },
                'multi_scale_3d_strides' + rot_num_id: {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })

        return batch_dict





class VirConv8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size,  **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.x_trans_train = X_TRANS()

        self.return_num_features_as_dict = model_cfg.RETURN_NUM_FEATURES_AS_DICT
        self.out_features = model_cfg.OUT_FEATURES
        self.layer_discard_rate = model_cfg.LAYER_DISCARD_RATE

        num_filters = model_cfg.NUM_FILTERS

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, num_filters[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(num_filters[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(num_filters[0], num_filters[0], 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(num_filters[0], num_filters[1], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2',
                  conv_type='spconv'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(num_filters[1], num_filters[1], 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(num_filters[1], num_filters[2], 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3',
                  conv_type='spconv'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(num_filters[2], num_filters[2], 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(num_filters[2], num_filters[3], 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4',
                  conv_type='spconv'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(num_filters[3], num_filters[3], 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(num_filters[3], self.out_features, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(self.out_features),
            nn.ReLU(),
        )
        if self.model_cfg.get('MM', False):
            self.vir_conv1 = NRConvBlock(input_channels, num_filters[0], stride=1, indice_key='vir1')
            self.vir_conv2 = NRConvBlock(num_filters[0], num_filters[1], stride=2, indice_key='vir2')
            self.vir_conv3 = NRConvBlock(num_filters[1], num_filters[2], stride=2, indice_key='vir3')
            self.vir_conv4 = NRConvBlock(num_filters[2], num_filters[3], stride=2, padding=(0, 1, 1),
                                          indice_key='vir4')

        self.num_point_features = self.out_features

        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features
        # for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False

    def decompose_tensor(self, tensor, i, batch_size):
        """
        decompose a sparse tensor by the given transformation index.
        """

        input_shape = tensor.spatial_shape[2]
        begin_shape_ids = i * (input_shape // 4)
        end_shape_ids = (i + 1) * (input_shape // 4)
        x_conv3_features = tensor.features
        x_conv3_coords = tensor.indices

        mask = (begin_shape_ids < x_conv3_coords[:, 3]) & (x_conv3_coords[:, 3] < end_shape_ids)
        this_conv3_feat = x_conv3_features[mask]
        this_conv3_coords = x_conv3_coords[mask]
        this_conv3_coords[:, 3] -= i * (input_shape // 4)
        this_shape = [tensor.spatial_shape[0], tensor.spatial_shape[1], tensor.spatial_shape[2] // 4]

        this_conv3_tensor = spconv.SparseConvTensor(
            features=this_conv3_feat,
            indices=this_conv3_coords.int(),
            spatial_shape=this_shape,
            batch_size=batch_size
        )
        return this_conv3_tensor

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        batch_size = batch_dict['batch_size']
        all_lidar_feat = []
        all_lidar_coords = []

        new_shape = [self.sparse_shape[0], self.sparse_shape[1], self.sparse_shape[2] * 4]

        if self.training:
            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                voxel_features, voxel_coords = batch_dict['voxel_features' + rot_num_id], batch_dict[
                    'voxel_coords' + rot_num_id]

                batch_size = batch_dict['batch_size']
                input_sp_tensor = spconv.SparseConvTensor(
                    features=voxel_features,
                    indices=voxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                x = self.conv_input(input_sp_tensor)

                x_conv1 = self.conv1(x)
                x_conv2 = self.conv2(x_conv1)
                x_conv3 = self.conv3(x_conv2)
                x_conv4 = self.conv4(x_conv3)

                # for detection head
                # [200, 176, 5] -> [200, 176, 2]
                out = self.conv_out(x_conv4)

                batch_dict.update({
                    'encoded_spconv_tensor' + rot_num_id: out,
                    'encoded_spconv_tensor_stride' + rot_num_id: 8,
                })
                batch_dict.update({
                    'multi_scale_3d_features' + rot_num_id: {
                        'x_conv1': x_conv1,
                        'x_conv2': x_conv2,
                        'x_conv3': x_conv3,
                        'x_conv4': x_conv4,
                    },
                    'multi_scale_3d_strides' + rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })
        else:
            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                voxel_features, voxel_coords = batch_dict['voxel_features' + rot_num_id], batch_dict[
                    'voxel_coords' + rot_num_id]

                # serial processing to parallel processing for speeding up inference
                all_lidar_feat.append(voxel_features)
                new_coord = voxel_coords.clone()
                new_coord[:, 3] += i * self.sparse_shape[2]
                all_lidar_coords.append(new_coord)

            all_lidar_feat = torch.cat(all_lidar_feat, 0)
            all_lidar_coords = torch.cat(all_lidar_coords)

            input_sp_tensor = spconv.SparseConvTensor(
                features=all_lidar_feat,
                indices=all_lidar_coords.int(),
                spatial_shape=new_shape,
                batch_size=batch_size
            )
            x = self.conv_input(input_sp_tensor)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            # for detection head
            # [200, 176, 5] -> [200, 176, 2]
            out = self.conv_out(x_conv4)

            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                this_conv3 = self.decompose_tensor(x_conv3, i, batch_size)
                this_conv4 = self.decompose_tensor(x_conv4, i, batch_size)
                this_out = self.decompose_tensor(out, i, batch_size)

                batch_dict.update({
                    'encoded_spconv_tensor' + rot_num_id: this_out,
                    'encoded_spconv_tensor_stride' + rot_num_id: 8,
                })
                batch_dict.update({
                    'multi_scale_3d_features' + rot_num_id: {
                        'x_conv1': None,
                        'x_conv2': None,
                        'x_conv3': this_conv3,
                        'x_conv4': this_conv4,
                    },
                    'multi_scale_3d_strides' + rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        for i in range(rot_num):
            if i == 0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)
            if self.model_cfg.get('MM', False):
                newvoxel_features, newvoxel_coords = batch_dict['voxel_features_mm' + rot_num_id], batch_dict[
                    'voxel_coords_mm' + rot_num_id]

                newinput_sp_tensor = spconv.SparseConvTensor(
                    features=newvoxel_features,
                    indices=newvoxel_coords.int(),
                    spatial_shape=self.sparse_shape,
                    batch_size=batch_size
                )
                if self.training:
                    layer_voxel_discard(newinput_sp_tensor, self.layer_discard_rate)

                calib = batch_dict['calib']
                batch_size = batch_dict['batch_size']
                if 'aug_param' in batch_dict:
                    trans_param = batch_dict['aug_param']
                else:
                    trans_param = None
                if 'transform_param' in batch_dict:
                    trans_param = batch_dict['transform_param'][:, i, :]

                newx_conv1 = self.vir_conv1(newinput_sp_tensor, batch_size, calib, 1, self.x_trans_train, trans_param)

                if self.training:
                    layer_voxel_discard(newx_conv1, self.layer_discard_rate)

                newx_conv2 = self.vir_conv2(newx_conv1, batch_size, calib, 2, self.x_trans_train, trans_param)

                if self.training:
                    layer_voxel_discard(newx_conv2, self.layer_discard_rate)

                newx_conv3 = self.vir_conv3(newx_conv2, batch_size, calib, 4, self.x_trans_train, trans_param)

                if self.training:
                    layer_voxel_discard(newx_conv3, self.layer_discard_rate)

                newx_conv4 = self.vir_conv4(newx_conv3, batch_size, calib, 8, self.x_trans_train, trans_param)

                batch_dict.update({
                    'encoded_spconv_tensor_stride_mm' + rot_num_id: 8
                })
                batch_dict.update({
                    'multi_scale_3d_features_mm' + rot_num_id: {
                        'x_conv1': newx_conv1,
                        'x_conv2': newx_conv2,
                        'x_conv3': newx_conv3,
                        'x_conv4': newx_conv4,
                    },
                    'multi_scale_3d_strides' + rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })

        return batch_dict

