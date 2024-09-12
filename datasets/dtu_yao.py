from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import *
from torchvision import transforms
import random

# the DTU dataset preprocessed by Yao Yao (only for training)
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews,  resize_wh=(800, 600), crop_wh =(640, 512),
                 ndepths=128, interval_scale=1.5, robust_train=False, random_crop=False):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.resize_wh = [int(v) for v in resize_wh.split(',')]
        self.crop_wh = [int(v) for v in crop_wh.split(',')]
        self.interval_scale = interval_scale
        self.robust_train = robust_train
        self.random_crop = random_crop

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = [line.rstrip() for line in f.readlines()]
        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(7):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        depth_max = depth_min + depth_interval * (self.ndepths - 1)
        return intrinsics, extrinsics, depth_min, depth_interval, depth_max

    def scale_mvs_input(self, depth_filename, mask_filename, resize_wh, crop_wh, scale):

        depth = np.array(read_pfm(depth_filename)[0], dtype=np.float32) * scale
        mask = np.array(Image.open(mask_filename), dtype=np.float32)
        mask = (mask > 10).astype(np.float32)

        h, w = depth.shape
        if h != int(resize_wh[1]) or w != int(resize_wh[0]):

            if depth_filename is not None:
                depth = cv2.resize(depth, (int(resize_wh[0]), int(resize_wh[1])))
                mask = cv2.resize(mask, (int(resize_wh[0]), int(resize_wh[1])))

        if int(crop_wh[1]) != int(resize_wh[1]) or int(crop_wh[0]) != int(resize_wh[0]):
            start_w = (int(resize_wh[0]) - int(crop_wh[0])) // 2
            start_h = (int(resize_wh[1]) - int(crop_wh[1])) // 2

            if depth_filename is not None:
                depth = np.copy(depth)[start_h: start_h + int(crop_wh[1]), start_w: start_w + int(crop_wh[0])]
                mask = np.copy(mask)[start_h: start_h + int(crop_wh[1]), start_w: start_w + int(crop_wh[0])]

        h, w = depth.shape
        depth_lr_ms = {"stage1": cv2.resize(depth, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
                       "stage2": cv2.resize(depth, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
                       "stage3": cv2.resize(depth, (w//2, h//2), interpolation=cv2.INTER_NEAREST)}
        mask_lr_ms = {"stage1": cv2.resize(mask, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
                      "stage2": cv2.resize(mask, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
                      "stage3": cv2.resize(mask, (w//2, h//2), interpolation=cv2.INTER_NEAREST)}
        return depth_lr_ms, mask_lr_ms

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        if self.robust_train:
            num_src_views = len(src_views)
            index = random.sample(range(num_src_views), self.nviews-1)
            view_ids = [ref_view] + [src_views[i] for i in index]
            #scale = random.uniform(0.8, 1.25)
            scale = 1
        else:
            view_ids = [ref_view] + src_views[:self.nviews - 1]
            scale = 1

        imgs = []
        mask = None
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,
                                        'Rectified/{}_train/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))

            mask_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))

            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)

            intrinsics, extrinsics, depth_min_, depth_interval_, depth_max_ = self.read_cam_file(proj_mat_filename)

            img = cv2.imread(img_filename)
            img = image_net_center(img)

            if i == 0:  # reference view
                depth_ms, mask = self.scale_mvs_input(depth_filename_hr, mask_filename_hr,
                                                                               self.resize_wh, self.crop_wh, scale)

                depth_min = depth_min_ * scale
                depth_interval = depth_interval_ * scale
                depth_max = depth_max_ * scale

            extrinsics[:3,3] *= scale
            intrinsics[:2,:] *= 4
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

            imgs.append(img)

        #all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        #ms proj_mats
        proj_matrices = np.stack(proj_matrices)
        stage1_pjmats = proj_matrices.copy()
        stage1_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.125
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.25
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 0.5

        proj_matrices_ms = {
            "stage1": stage1_pjmats,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }

        return {"imgs": imgs,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "depth_min": depth_min,
                "depth_interval": depth_interval,
                "depth_max": depth_max,
                "mask": mask}
