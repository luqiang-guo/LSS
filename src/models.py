"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18

from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        # x1 上采样  + x2
        # conv
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")
        print(self.trunk)

        self.up1 = Up(320+112, 512)

        # 没有输出DxC维度的特征，通过两个向量外积的方式得到一个矩阵
        # 很像是特征值特征向量
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x): 
        # 提取efficient net 提取特征     # 5 3 128 352   nchw
        x = self.get_eff_depth(x)       # 5 512 8 22    nchw
        # Depth
        # 得到D+C的特征
        x = self.depthnet(x)            # 5 105 8 22    n (d + c)hw

        depth = self.get_depth_dist(x[:, :self.D])  # 5 41 8 22    n(d)hw
        # 矩阵的外积得到特征矩阵
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2) # 5 64 41 8 22

        '''
        个人感悟：
        这里原本可以直接通过nn学出来一个输出的特征矩阵,
        感觉这里学出来的是矩阵谱分解后的一层，不知道这样子做的目的？
        '''
        return depth, new_x

    def get_eff_depth(self, x):
        '''
        使用efficient net 提取输入x的特征
        通过UP 进行多尺度融合最终得到输出的特征
        '''

        # 下面的链接不知道缘由：
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # 为了获取中间特征，人为的展开efficient
        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x

        print('________________________________________________________')
        print("x size -> ", x.size())
        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            print("x size -> ", x.size())
            if prev_x.size(2) > x.size(2):
                print('eff block', prev_x.size(), x.size())
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x
        # 多尺度特征融合
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x) # 5 64 41 8 22
        
        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True
    
    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']    # 128 352
        fH, fW = ogfH // self.downsample, ogfW // self.downsample # 8 22
        # grid_conf['dbound'] [4 45 1.0]
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW) 
        # [41] -> [41, 1, 1] -> [41, 8, 22]
        # ds [4]
        # [[4 4 ...  4 4]
        #  [4 4 ...  4 4]]
        # ...
        # [[45 45 ...  45 45]
        #  [45 45 ...  45 45]]

        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1) # 41 8 22 3
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """

        '''
        mini nuScen
        rots        [1, 5, 3, 3]    旋转
        trans       [1, 5, 3]       平移
        intrins     [1, 5, 3, 3]    内参
        post_rots   [1, 5, 3, 3]    外参
        post_trans  [1, 5, 3]       图像增强
        '''

        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        #
        points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        '''
        x            shape ->  [4, 5, 3, 128, 352]
        return       shape ->  [4, 5, 41, 8, 22, 64]
        '''

        B, N, C, imH, imW = x.shape

        # pytorch 只能处理4维 所以要view下
        x = x.view(B*N, C, imH, imW)
        # out x shape :  [5, 3, 128, 352]

        x = self.camencode(x) # 5 64 41 8 22
        # out x shape :  [5, 64, 41, 8, 22]
        # camC  这是什么参数 downsample下采样
        x = x.view(B, N, self.camC, self.D, imH//self.downsample, imW//self.downsample)
        # out x shape :  [1, 5, 64, 41, 8, 22]

        # 把channel 最后的目的是？
        x = x.permute(0, 1, 3, 4, 5, 2)
        # out x shape :  [1, 5, 41, 8, 22, 64]  b n d h w camc

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape  # [1, 5, 41, 8, 22, 64]  b n d h w camc
        Nprime = B*N*D*H*W

        # flatten x
        x = x.reshape(Nprime, C) # 36080 64

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx/2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime//B, 1], ix,
                             device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0])\
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1])\
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B)\
            + geom_feats[:, 1] * (self.nx[2] * B)\
            + geom_feats[:, 2] * B\
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x, rots, trans, intrins, post_rots, post_trans):
        '''
        x           [4, 5, 3, 128, 352]
        rots        [4, 5, 3, 3]
        trans       [4, 5, 3]
        intrins     [4, 5, 3, 3]
        post_rots   [4, 5, 3, 3]
        post_trans  [4, 5, 3]
        '''
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans) # 1 5 41 8 22
        x = self.get_cam_feats(x) # [1, 5, 41, 8, 22, 64]  b n d h w camc

        x = self.voxel_pooling(geom, x) 

        return x

    def forward(self, x, rots, trans, intrins, post_rots, post_trans):
        '''
        x           [4, 5, 3, 128, 352]
        rots        [4, 5, 3, 3]
        trans       [4, 5, 3]
        intrins     [4, 5, 3, 3]
        post_rots   [4, 5, 3, 3]
        post_trans  [4, 5, 3]
        '''

        x = self.get_voxels(x, rots, trans, intrins, post_rots, post_trans)
        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
