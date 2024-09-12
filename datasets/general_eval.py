from torch.utils.data import Dataset
import numpy as np
import os, cv2, time
from PIL import Image
from datasets.data_io import *

class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, resize_wh=(768,576), crop_wh=(640, 512),
                 ndepths=128, interval_scale=1):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.resize_wh = [int(v) for v in resize_wh.split(',')]
        self.crop_wh = [int(v) for v in crop_wh.split(',')]
        self.interval_scale = interval_scale

        assert self.mode == "test"
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        scans = self.listfile
        # scans
        for scan in scans:
            pair_file = "{}/pair.txt".format(scan)
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    if len(src_views) != 0:
                        metas.append((scan, ref_view, src_views))

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
        depth_interval = float(lines[11].split()[1])

        if len(lines[11].split()) >= 4:
            # num_depth = lines[11].split()[2]
            depth_max = float(lines[11].split()[3])
            depth_interval = (depth_max - depth_min) / (self.ndepths - 1)
        else:
            depth_interval = depth_interval * self.interval_scale
            depth_max = depth_min + depth_interval * (self.ndepths - 1)

        return intrinsics, extrinsics, depth_min, depth_interval, depth_max

    def scale_mvs_input(self, img_filename, intrinsics, resize_wh, crop_wh):

        img = cv2.imread(img_filename)
        img = image_net_center(img)

        h, w = img.shape[:2]

        if h != int(resize_wh[1]) or w != int(resize_wh[0]):
            img = cv2.resize(img, (int(resize_wh[0]), int(resize_wh[1])))

            scale_w = 1.0 * int(resize_wh[0]) / w
            scale_h = 1.0 * int(resize_wh[1]) / h
            intrinsics[0, :] *= scale_w
            intrinsics[1, :] *= scale_h

        if int(crop_wh[1]) != int(resize_wh[1]) or int(crop_wh[0]) != int(resize_wh[0]):
            start_w = (int(resize_wh[0]) - int(crop_wh[0])) // 2
            start_h = (int(resize_wh[1]) - int(crop_wh[1])) // 2
            img = np.copy(img)[start_h: start_h + int(crop_wh[1]), start_w: start_w + int(crop_wh[0])]

            new_intrinsics = intrinsics
            new_intrinsics[0,2] = intrinsics[0,2] - start_w
            new_intrinsics[1,2] = intrinsics[1,2] - start_h
            intrinsics = new_intrinsics

        return img, intrinsics

    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        view_ids = [ref_view] + src_views[:self.nviews - 1]

        imgs = []
        proj_matrices = []

        for i, vid in enumerate(view_ids):
            img_filename = os.path.join(self.datapath, '{}/images/{:0>8}.jpg'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, '{}/cams/{:0>8}_cam.txt'.format(scan, vid))

            if i == 0:
                intrinsics, extrinsics, depth_min, depth_interval, depth_max = self.read_cam_file(proj_mat_filename)
            else:
                intrinsics, extrinsics, _, _, _ = self.read_cam_file(proj_mat_filename)

            # scale input
            img, intrinsics = self.scale_mvs_input(img_filename, intrinsics, self.resize_wh, self.crop_wh)

            imgs.append(img)
            # extrinsics, intrinsics
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
            proj_mat[0, :4, :4] = extrinsics
            proj_mat[1, :3, :3] = intrinsics

            proj_matrices.append(proj_mat)

        # all
        imgs = np.stack(imgs).transpose([0, 3, 1, 2])

        # ms proj_mats
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
                "depth_min": depth_min,
                "depth_interval": depth_interval,
                "depth_max": depth_max,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}"}
