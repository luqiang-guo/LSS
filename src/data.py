"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import os
import numpy as np
from PIL import Image
import cv2
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import Box
from glob import glob

from .tools import get_lidar_data, img_transform, normalize_img, gen_dx_bx


class NuscData(torch.utils.data.Dataset):
    def __init__(self, nusc, is_train, data_aug_conf, grid_conf):
        '''
        data_aug_conf : {
            'resize_lim': (0.193, 0.225), 'final_dim': (128, 352), 
            'rot_lim': (-5.4, 5.4), 'H': 900, 'W': 1600, 'rand_flip': True, 'bot_pct_lim': (0.0, 0.22), 
            'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'], 
            'Ncams': 5}

        grid_conf : {
            'xbound': [-50.0, 50.0, 0.5], 
            'ybound': [-50.0, 50.0, 0.5], 
            'zbound': [-10.0, 10.0, 20.0], 
            'dbound': [4.0, 45.0, 1.0]}
        '''
        self.nusc = nusc
        self.is_train = is_train
        self.data_aug_conf = data_aug_conf
        self.grid_conf = grid_conf

        # 训练或验证场景
        self.scenes = self.get_scenes()
        
        # 重新整理sample  按照场景和时间戳排序
        # ixes =  sample
        self.ixes = self.prepro()

        # dx : tensor([ 0.5000,  0.5000, 20.0000]) step ?
        # bx : tensor([-49.7500, -49.7500,   0.0000]) start ?
        # nx : tensor([200, 200,   1]) num step ?
        dx, bx, nx = gen_dx_bx(grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])
 
        self.dx, self.bx, self.nx = dx.numpy(), bx.numpy(), nx.numpy()

        # 验证数据集
        self.fix_nuscenes_formatting()

        print(self)

    def fix_nuscenes_formatting(self):
        """If nuscenes is stored with trainval/1 trainval/2 ... structure, adjust the file paths
        stored in the nuScenes object.
        """
        # check if default file paths work
        rec = self.ixes[0]
        sampimg = self.nusc.get('sample_data', rec['data']['CAM_FRONT'])
        imgname = os.path.join(self.nusc.dataroot, sampimg['filename'])

        # sampimg ->  {'token': '524d443c501a4f98a14508c3fb6f6de3', 
        # 'sample_token': 'b5989651183643369174912bc5641d3b', 
        # 'ego_pose_token': '524d443c501a4f98a14508c3fb6f6de3', 
        # 'calibrated_sensor_token': '1b10ebeec99e4608977957c3f04c996d', 
        # 'timestamp': 1538984233512470, 
        # 'fileformat': 'jpg', 
        # 'is_key_frame': True, 
        # 'height': 900, 
        # 'width': 1600, 
        # 'filename': 'samples/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984233512470.jpg', 
        # 'prev': '', 
        # 'next': 'bfb50929fee1418398db9875dd5f8108', 
        # 'sensor_modality': 'camera', 
        # 'channel': 'CAM_FRONT'}
        # imgname ->  /home/test/dataset/nuscenes/mini/samples/CAM_FRONT/n015-2018-10-08-15-36-50+0800__CAM_FRONT__1538984233512470.jpg

        def find_name(f):
            d, fi = os.path.split(f)
            d, di = os.path.split(d)
            d, d0 = os.path.split(d)
            d, d1 = os.path.split(d)
            d, d2 = os.path.split(d)
            return di, fi, f'{d2}/{d1}/{d0}/{di}/{fi}'

        # adjust the image paths if needed
        if not os.path.isfile(imgname):
            print('adjusting nuscenes file paths')
            fs = glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/CAM*/*.jpg'))
            fs += glob(os.path.join(self.nusc.dataroot, 'samples/*/samples/LIDAR_TOP/*.pcd.bin'))
            info = {}
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'samples/{di}/{fi}'] = fname
            fs = glob(os.path.join(self.nusc.dataroot, 'sweeps/*/sweeps/LIDAR_TOP/*.pcd.bin'))
            for f in fs:
                di, fi, fname = find_name(f)
                info[f'sweeps/{di}/{fi}'] = fname
            for rec in self.nusc.sample_data:
                if rec['channel'] == 'LIDAR_TOP' or (rec['is_key_frame'] and rec['channel'] in self.data_aug_conf['cams']):
                    rec['filename'] = info[rec['filename']]

    
    def get_scenes(self):
        # filter by scene split
        split = {
            'v1.0-trainval': {True: 'train', False: 'val'},
            'v1.0-mini': {True: 'mini_train', False: 'mini_val'},
        }[self.nusc.version][self.is_train]

        # 该函数用于根据指定的规则将数据集划分为训练集、验证集和测试集，并在每个集合中创建场景列表。
        # 场景是数据集中连续的时间序列，通常对应于车辆在特定区域内的一系列行驶轨迹。
        scenes = create_splits_scenes()[split]

        return scenes

    def prepro(self):

        # 每个 sample
        # {'token': 'ca9a282c9e77460f8360f564131a8af5', 
        #  'timestamp': 1532402927647951, 
        #  'prev': '', 
        #  'next': '39586f9d59004284a7114a68825e8eec', 
        #  'scene_token': 'cc8c0bf57f984915a77078b10eb33198', 
        #  'data': {'RADAR_FRONT': '37091c75b9704e0daa829ba56dfa0906', 
        #           'RADAR_FRONT_LEFT': '11946c1461d14016a322916157da3c7d', 
        #           'RADAR_FRONT_RIGHT': '491209956ee3435a9ec173dad3aaf58b', 
        #           'RADAR_BACK_LEFT': '312aa38d0e3e4f01b3124c523e6f9776', 
        #           'RADAR_BACK_RIGHT': '07b30d5eb6104e79be58eadf94382bc1', 
        #           'LIDAR_TOP': '9d9bf11fb0e144c8b446d54a8a00184f', 
        #           'CAM_FRONT': 'e3d495d4ac534d54b321f50006683844', 
        #           'CAM_FRONT_RIGHT': 'aac7867ebf4f446395d29fbd60b63b3b', 
        #           'CAM_BACK_RIGHT': '79dbb4460a6b40f49f9c150cb118247e', 
        #           'CAM_BACK': '03bea5763f0f4722933508d5999c5fd8', 
        #           'CAM_BACK_LEFT': '43893a033f9c46d4a51b5e08a67a1eb7', 
        #           'CAM_FRONT_LEFT': 'fe5422747a7d4268a4b07fc396707b23'}, 
        #           'anns': ['ef63a697930c4b20a6b9791f423351da', 
        #                    '6b89da9bf1f84fd6a5fbe1c3b236f809', 
        #                    '924ee6ac1fed440a9d9e3720aac635a0', 
        #                    '91e3608f55174a319246f361690906ba', 
        #                    '...',
        #                    '15a3b4d60b514db5a3468e2aef72a90c', 
        #                    '18cc2837f2b9457c80af0761a0b83ccc', 
        #                    '2bfcc693ae9946daba1d9f2724478fd4']}
        # 把所有的 samp 组成一个list
        samples = [samp for samp in self.nusc.sample]

        # remove samples that aren't in this split
        samples = [samp for samp in samples if
                   self.nusc.get('scene', samp['scene_token'])['name'] in self.scenes]

        # 优先按照场景排序，每个场景按照时间戳排序
        # sort by scene, timestamp (only to make chronological viz easier)
        samples.sort(key=lambda x: (x['scene_token'], x['timestamp']))

        return samples
    
    def sample_augmentation(self):

        # 'resize_lim': (0.193, 0.225), 
        # 'final_dim': (128, 352), 
        # 'rot_lim': (-5.4, 5.4), 
        # 'H': 900, 'W': 1600
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        # 128, 352
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            # 创建resize  crop
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH/H, fW/W)
            resize_dims = (int(W*resize), int(H*resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf['bot_pct_lim']))*newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_image_data(self, rec, cams):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        for cam in cams:
            # 从sample中选取一张图片
            samp = self.nusc.get('sample_data', rec['data'][cam])
            # 获取图片绝对路径
            imgname = os.path.join(self.nusc.dataroot, samp['filename'])
            # 解码图片获取rgb数据
            img = Image.open(imgname)
            # post_rot 单位矩阵
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            # 校准sensor 什么呢？
            # sens -> {
            # 'token': '5415fa359d294928be3fbd7810b3b682', 
            # 'sensor_token': 'a89643a5de885c6486df2232dc954da2', 
            # 'translation': [1.04852047718, 0.483058131052, 1.56210154484], 
            # 'rotation': [0.7048620297871717, -0.6907306801461466, -0.11209091960167808, 0.11617345743327073], 
            # 'camera_intrinsic': [[1254.9860565800168, 0.0, 829.5769333630991], [0.0, 1254.9860565800168, 467.1680561863987], [0.0, 0.0, 1.0]]}
            sens = self.nusc.get('calibrated_sensor', samp['calibrated_sensor_token'])
            print('sens ->', sens)
            #获取相机信息？

            # intrin -> tensor([[1.2550e+03, 0.0000e+00, 8.2958e+02],
            #         [0.0000e+00, 1.2550e+03, 4.6717e+02],
            #         [0.0000e+00, 0.0000e+00, 1.0000e+00]])
            intrin = torch.Tensor(sens['camera_intrinsic'])
            print('intrin ->', intrin)
            # rotation 
            # rot -> tensor([[ 0.9479, -0.0089, -0.3185],
            #         [ 0.3186,  0.0188,  0.9477],
            #         [-0.0025, -0.9998,  0.0207]])
            rot = torch.Tensor(Quaternion(sens['rotation']).rotation_matrix)

            # tran -> tensor([1.0485, 0.4831, 1.5621])
            tran = torch.Tensor(sens['translation'])

            # 计算图像增强的参数
            # augmentation (resize, crop, horizontal flip, rotate)
            # resize        :   0.21148668003737284
            # resize_dims   :   (338, 190)
            # crop          :   (0, 30, 352, 158)
            # flip          :   False
            # rotate        :   -0.9546193864738521
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation()

            # 图像变换
            img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                     resize=resize,
                                                     resize_dims=resize_dims,
                                                     crop=crop,
                                                     flip=flip,
                                                     rotate=rotate,
                                                     )
            
            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            # normalize
            imgs.append(normalize_img(img))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        return (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                torch.stack(intrins), torch.stack(post_rots), torch.stack(post_trans))

    def get_lidar_data(self, rec, nsweeps):
        pts = get_lidar_data(self.nusc, rec,
                       nsweeps=nsweeps, min_distance=2.2)
        return torch.Tensor(pts)[:3]  # x,y,z

    def get_binimg(self, rec):
        egopose = self.nusc.get('ego_pose',
                                self.nusc.get('sample_data', rec['data']['LIDAR_TOP'])['ego_pose_token'])
        trans = -np.array(egopose['translation'])
        rot = Quaternion(egopose['rotation']).inverse
        img = np.zeros((self.nx[0], self.nx[1]))
        for tok in rec['anns']:
            inst = self.nusc.get('sample_annotation', tok)
            # add category for lyft
            if not inst['category_name'].split('.')[0] == 'vehicle':
                continue
            box = Box(inst['translation'], inst['size'], Quaternion(inst['rotation']))
            box.translate(trans)
            box.rotate(rot)

            pts = box.bottom_corners()[:2].T
            pts = np.round(
                (pts - self.bx[:2] + self.dx[:2]/2.) / self.dx[:2]
                ).astype(np.int32)
            pts[:, [1, 0]] = pts[:, [0, 1]]
            cv2.fillPoly(img, [pts], 1.0)

        return torch.Tensor(img).unsqueeze(0)

    def choose_cams(self):
        '''
        根据Ncams, 从环视的相机的中选取图片。
        '''
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
        return cams

    def __str__(self):
        return f"""NuscData: {len(self)} samples. Split: {"train" if self.is_train else "val"}.
                   Augmentation Conf: {self.data_aug_conf}"""

    def __len__(self):
        return len(self.ixes)


class VizData(NuscData):
    def __init__(self, *args, **kwargs):
        super(VizData, self).__init__(*args, **kwargs)
    
    def __getitem__(self, index):
        # 从samples 中选取一个sample
        rec = self.ixes[index]
        
        # 根据 Ncams，选取投影相机位置。
        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        lidar_data = self.get_lidar_data(rec, nsweeps=3)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, lidar_data, binimg


class SegmentationData(NuscData):
    def __init__(self, *args, **kwargs):
        super(SegmentationData, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        rec = self.ixes[index]

        cams = self.choose_cams()
        imgs, rots, trans, intrins, post_rots, post_trans = self.get_image_data(rec, cams)
        binimg = self.get_binimg(rec)
        
        return imgs, rots, trans, intrins, post_rots, post_trans, binimg


def worker_rnd_init(x):
    np.random.seed(13 + x)


def compile_data(version, dataroot, data_aug_conf, grid_conf, bsz,
                 nworkers, parser_name):

    '''
    version : mini
    dataroot : /home/test/dataset/nuscenes
    data_aug_conf : {
        'resize_lim': (0.193, 0.225), 'final_dim': (128, 352), 
        'rot_lim': (-5.4, 5.4), 'H': 900, 'W': 1600, 'rand_flip': True, 'bot_pct_lim': (0.0, 0.22), 
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'], 
        'Ncams': 5}

    grid_conf : {
        'xbound': [-50.0, 50.0, 0.5], 
        'ybound': [-50.0, 50.0, 0.5], 
        'zbound': [-10.0, 10.0, 20.0], 
        'dbound': [4.0, 45.0, 1.0]}
    bsz : 4
    nworkers : 10
    parser_name : segmentationdata
    '''
    # 实例化nuScenes
    nusc = NuScenes(version='v1.0-{}'.format(version),
                    dataroot=os.path.join(dataroot, version),
                    verbose=False)
    parser = {
        'vizdata': VizData,
        'segmentationdata': SegmentationData,
    }[parser_name]
    traindata = parser(nusc, is_train=True, data_aug_conf=data_aug_conf,
                         grid_conf=grid_conf)
    valdata = parser(nusc, is_train=False, data_aug_conf=data_aug_conf,
                       grid_conf=grid_conf)

    trainloader = torch.utils.data.DataLoader(traindata, batch_size=bsz,
                                              shuffle=True,
                                              num_workers=nworkers,
                                              drop_last=True,
                                              worker_init_fn=worker_rnd_init)
    valloader = torch.utils.data.DataLoader(valdata, batch_size=bsz,
                                            shuffle=False,
                                            num_workers=nworkers)

    return trainloader, valloader
