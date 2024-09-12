from torch.utils.data import Dataset
import numpy as np
import os, cv2, time, math
from PIL import Image
from datasets.data_io import *
import torch
from torchvision import transforms
import random

class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, resize_wh=(768,576), crop_wh=(640, 512),
                 ndepths=128, interval_scale=1, robust_train=False, random_crop=False):
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
        self.color_augment = transforms.ColorJitter(brightness=0.5, contrast=0.5)

        self.scan_factors = {}

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = [line.rstrip() for line in f.readlines()]

        # scans
        for scan in scans:
            pair_file = "{}/cams/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # filter by no src view and fill to nviews
                    if len(src_views) > 0:
                        if len(src_views) < self.nviews:
                            print(f'sample {scan} id {view_idx} does not have enough sources, total src view {len(src_views)}')
                        else:
                            metas.append((scan, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, scan, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1])
        depth_num = float(lines[11].split()[2])
        depth_max = float(lines[11].split()[3])

        if scan not in self.scan_factors:
            self.scan_factors[scan] = 1.0
        depth_min *= self.scan_factors[scan]
        depth_max *= self.scan_factors[scan]
        depth_interval *= self.scan_factors[scan]
        extrinsics[:3,3] *= self.scan_factors[scan]
        return intrinsics, extrinsics, depth_min, depth_interval, depth_max

    def scale_mvs_input(self, scan, img_filename, intrinsics, depth_filename, depth_para, resize_wh, crop_wh, scale):

        img = cv2.imread(img_filename)
        if self.mode == "train":
            img = random_contrast(img, strength_range=[0.3, 1.5])
            img = random_brightness(img, max_abs_change=50)
            img = motion_blur(img, max_kernel_size=3)
        img = image_net_center(img)

        if depth_filename is not None:
            depth = np.array(read_pfm(depth_filename)[0], dtype=np.float32) * scale
            depth = depth * self.scan_factors[scan]
            mask = (depth>=depth_para[0])&(depth<=depth_para[1])
            mask = mask.astype(np.float32)

        h, w = img.shape[:2]
        if self.random_crop:
            if int(crop_wh[1]) != h or int(crop_wh[0]) != w:
                start_w = int(self.crop_wh[2])
                start_h = int(self.crop_wh[3])
                img = np.copy(img)[start_h: start_h + int(crop_wh[1]), start_w: start_w + int(crop_wh[0])]
                new_intrinsics = intrinsics
                new_intrinsics[0,2] = intrinsics[0,2] - start_w
                new_intrinsics[1,2] = intrinsics[1,2] - start_h
                intrinsics = new_intrinsics

                if depth_filename is not None:
                    depth = np.copy(depth)[start_h: start_h + int(crop_wh[1]), start_w: start_w + int(crop_wh[0])]
                    mask = np.copy(mask)[start_h: start_h + int(crop_wh[1]), start_w: start_w + int(crop_wh[0])]

        else:
            if h != int(resize_wh[1]) or w != int(resize_wh[0]):
                img = cv2.resize(img, (int(resize_wh[0]), int(resize_wh[1])))

                scale_w = 1.0 * int(resize_wh[0]) / w
                scale_h = 1.0 * int(resize_wh[1]) / h
                intrinsics[0, :] *= scale_w
                intrinsics[1, :] *= scale_h

                if depth_filename is not None:
                    depth = cv2.resize(depth, (int(resize_wh[0]), int(resize_wh[1])))
                    mask = cv2.resize(mask, (int(resize_wh[0]), int(resize_wh[1])))

            if int(crop_wh[1]) != int(resize_wh[1]) or int(crop_wh[0]) != int(resize_wh[0]):
                start_w = (int(resize_wh[0]) - int(crop_wh[0])) // 2
                start_h = (int(resize_wh[1]) - int(crop_wh[1])) // 2
                img = np.copy(img)[start_h: start_h + int(crop_wh[1]), start_w: start_w + int(crop_wh[0])]

                new_intrinsics = intrinsics
                new_intrinsics[0,2] = intrinsics[0,2] - start_w
                new_intrinsics[1,2] = intrinsics[1,2] - start_h
                intrinsics = new_intrinsics

                if depth_filename is not None:
                    depth = np.copy(depth)[start_h: start_h + int(crop_wh[1]), start_w: start_w + int(crop_wh[0])]
                    mask = np.copy(mask)[start_h: start_h + int(crop_wh[1]), start_w: start_w + int(crop_wh[0])]


        if depth_filename is not None:
            h, w = depth.shape
            depth_lr_ms = {"stage1": cv2.resize(depth, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
                           "stage2": cv2.resize(depth, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
                           "stage3": cv2.resize(depth, (w//2, h//2), interpolation=cv2.INTER_NEAREST)}
            mask_lr_ms = {"stage1": cv2.resize(mask, (w//8, h//8), interpolation=cv2.INTER_NEAREST),
                          "stage2": cv2.resize(mask, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
                          "stage3": cv2.resize(mask, (w//2, h//2), interpolation=cv2.INTER_NEAREST)}
            return img, intrinsics, depth_lr_ms, mask_lr_ms
        else:
            return img, intrinsics


    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
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

        if self.random_crop:
            img = cv2.imread(os.path.join(self.datapath, '{}/blended_images/{:0>8}.jpg'.format(scan, view_ids[0])))
            h, w = img.shape[:2]
            start_w = random.randint(0, w - int(self.crop_wh[0]))
            start_h = random.randint(0, h - int(self.crop_wh[1]))
            self.crop_wh.append(start_w)
            self.crop_wh.append(start_h)

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath, '{}/blended_images/{:0>8}.jpg'.format(scan, vid))
            depth_filename_hr = os.path.join(self.datapath, '{}/rendered_depth_maps/{:0>8}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt').format(scan, vid)

            intrinsics, extrinsics, depth_min_, depth_interval_, depth_max_ = self.read_cam_file(scan, proj_mat_filename)

            if i == 0:  # reference view
                depth_min = depth_min_ * scale
                depth_interval = depth_interval_ * scale
                depth_max = depth_max_ * scale

                img, intrinsics, depth_ms, mask = self.scale_mvs_input(scan, img_filename, intrinsics, depth_filename_hr,
                                                                       [depth_min, depth_max], self.resize_wh, self.crop_wh, scale)

            else:
                img, intrinsics = self.scale_mvs_input(scan, img_filename, intrinsics,
                                                                       None, None, self.resize_wh,
                                                                       self.crop_wh, scale)

            extrinsics[:3, 3] *= scale
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
