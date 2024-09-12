import argparse, os, time, sys, gc, cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import numpy as np
from utils import *
from datasets.data_io import read_pfm
from plyfile import PlyData, PlyElement
from PIL import Image

from typing import List, Union, Dict
import open3d as o3d

parser = argparse.ArgumentParser(description='Depth filter and fuse')
parser.add_argument('--testpath', default='',help='testing data dir for some scenes')
parser.add_argument('--testlist', default='', help='testing scene list')
parser.add_argument('--gpu_device', type=str, default='2', help='gpu no.')

parser.add_argument('--outdir', default='./outputs', help='output dir')
parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')

parser.add_argument('--pthresh', type=str, default='.6,.6,.6')
parser.add_argument('--vthresh', type=int, default=11)
parser.add_argument('--downsample', type=float, default=None)
parser.add_argument('--dist_base', type=float, default=8.0)
parser.add_argument('--rel_diff_base', type=float, default=1300.0)
parser.add_argument('--write_mask', type=bool, default=True)

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1

# read intrinsics and extrinsics
def read_camera_parameters(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.rstrip() for line in lines]
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
    # TODO: assume the feature is 1/4 of the original image size
    # intrinsics[:2, :] /= 4
    return intrinsics, extrinsics

def recursive_apply(obj: Union[List, Dict], func):
    assert type(obj) == dict or type(obj) == list
    idx_iter = obj if type(obj) == dict else range(len(obj))
    for k in idx_iter:
        if type(obj[k]) == list or type(obj[k]) == dict:
            recursive_apply(obj[k], func)
        else:
            obj[k] = func(obj[k])

def load_pair(file: str, min_views: int=None):
    with open(file) as f:
        lines = f.readlines()
    n_cam = int(lines[0])
    pairs = {}
    img_ids = []
    img_ids_all = []
    for i in range(1, 1+2*n_cam, 2):
        pair = []
        score = []
        img_id = lines[i].strip()
        pair_str = lines[i+1].strip().split(' ')
        n_pair = int(pair_str[0])
        if n_pair == 0:
            print('pass',img_id)
            continue
        img_ids_all.append(img_id)
        #if min_views is not None and n_pair < min_views:
        if n_pair <= 1:
            print('pass',img_id)
            continue
        for j in range(1, 1+2*n_pair, 2):
            pair.append(pair_str[j])
            score.append(float(pair_str[j+1]))
        img_ids.append(img_id)
        pairs[img_id] = {'id': img_id, 'index': i//2, 'pair': pair, 'score': score}
    pairs['id_list'] = img_ids
    pairs['id_list_all'] = img_ids_all
    return pairs

def prob_filter(ref_probs, pthresh, greater=True):  # n3hw -> n1hw
    cmpr = lambda x, y: x > y if greater else lambda x, y: x < y
    masks = cmpr(ref_probs, torch.Tensor(pthresh).to(ref_probs.dtype).to(ref_probs.device).view(1,-1,1,1)).to(ref_probs.dtype)
    mask = (masks.sum(dim=1, keepdim=True) >= (len(pthresh)-0.1))
    return mask

def get_pixel_grids(height, width):
    x_coord = (torch.arange(width, dtype=torch.float32).cuda() + 0.5).repeat(height, 1)
    y_coord = (torch.arange(height, dtype=torch.float32).cuda() + 0.5).repeat(width, 1).t()
    ones = torch.ones_like(x_coord)
    indices_grid = torch.stack([x_coord, y_coord, ones], dim=-1).unsqueeze(-1)  # hw31
    return indices_grid

def idx_img2cam(idx_img_homo, depth, cam):  # nhw31, n1hw -> nhw41
    idx_cam = cam[:,1:2,:3,:3].unsqueeze(1).inverse() @ idx_img_homo  # nhw31
    idx_cam = idx_cam / (idx_cam[...,-1:,:]+1e-9) * depth.permute(0,2,3,1).unsqueeze(4)  # nhw31
    idx_cam_homo = torch.cat([idx_cam, torch.ones_like(idx_cam[...,-1:,:])], dim=-2)  # nhw41
    # FIXME: out-of-range is 0,0,0,1, will have valid coordinate in world
    return idx_cam_homo

def idx_cam2world(idx_cam_homo, cam):  # nhw41 -> nhw41
    idx_world_homo =  cam[:,0:1,...].unsqueeze(1).inverse() @ idx_cam_homo  # nhw41
    idx_world_homo = idx_world_homo / (idx_world_homo[...,-1:,:]+1e-9)  # nhw41
    return idx_world_homo

def idx_world2cam(idx_world_homo, cam):  # nhw41 -> nhw41
    idx_cam_homo = cam[:,0:1,...].unsqueeze(1) @ idx_world_homo  # nhw41
    idx_cam_homo = idx_cam_homo / (idx_cam_homo[...,-1:,:]+1e-9)   # nhw41
    return idx_cam_homo


def idx_cam2img(idx_cam_homo, cam):  # nhw41 -> nhw31
    idx_cam = idx_cam_homo[...,:3,:] / (idx_cam_homo[...,3:4,:]+1e-9)  # nhw31
    idx_img_homo = cam[:,1:2,:3,:3].unsqueeze(1) @ idx_cam  # nhw31
    idx_img_homo = idx_img_homo / (idx_img_homo[...,-1:,:]+1e-9)
    return idx_img_homo

def bin_op_reduce(lst: List, func):
    result = lst[0]
    for i in range(1, len(lst)):
        result = func(result, lst[i])
    return result

def project_img(src_img, dst_depth, src_cam, dst_cam, height=None, width=None):  # nchw, n1hw -> nchw, n1hw
    if height is None: height = src_img.size()[-2]
    if width is None: width = src_img.size()[-1]
    dst_idx_img_homo = get_pixel_grids(height, width).unsqueeze(0)  # nhw31
    dst2src_idx_img_homos = []
    for i in range(dst_depth.shape[0]):
        dst_idx_cam_homo = idx_img2cam(dst_idx_img_homo, dst_depth[i].unsqueeze(0), dst_cam[i].unsqueeze(0))
        dst_idx_world_homo = idx_cam2world(dst_idx_cam_homo, dst_cam[i].unsqueeze(0))
        dst2src_idx_cam_homo = idx_world2cam(dst_idx_world_homo, src_cam[i].unsqueeze(0))
        dst2src_idx_img_homo = idx_cam2img(dst2src_idx_cam_homo, src_cam[i].unsqueeze(0))
        dst2src_idx_img_homos.append(dst2src_idx_img_homo)
    dst2src_idx_img_homos = torch.cat(dst2src_idx_img_homos, 0)
    del dst_idx_cam_homo, dst_idx_world_homo, dst2src_idx_cam_homo, dst2src_idx_img_homo
    warp_coord = dst2src_idx_img_homos[..., :2, 0]  # nhw2
    warp_coord[..., 0] /= width
    warp_coord[..., 1] /= height
    warp_coord = (warp_coord * 2 - 1).clamp(-1.1, 1.1)  # nhw2
    in_range = bin_op_reduce(
        [-1 <= warp_coord[..., 0], warp_coord[..., 0] <= 1, -1 <= warp_coord[..., 1], warp_coord[..., 1] <= 1],
        torch.min).to(src_img.dtype).unsqueeze(1)  # n1hw
    warped_img = F.grid_sample(src_img, warp_coord, mode='bilinear', padding_mode='zeros', align_corners=False)
    return warped_img, in_range

def get_reproj(ref_depth, srcs_depth, ref_cam, srcs_cam):  # n1hw, nv1hw -> nv3hw, nv1hw
    n, v, _, h, w = srcs_depth.size()
    srcs_depth_f = srcs_depth.view(n * v, 1, h, w)
    srcs_valid_f = (srcs_depth_f > 1e-9).to(srcs_depth_f.dtype)
    srcs_cam_f = srcs_cam.view(n * v, 2, 4, 4)
    ref_depth_r = ref_depth.unsqueeze(1).repeat(1, v, 1, 1, 1).view(n * v, 1, h, w)
    ref_cam_r = ref_cam.unsqueeze(1).repeat(1, v, 1, 1, 1).view(n * v, 2, 4, 4)
    idx_img = get_pixel_grids(h, w).unsqueeze(0)  # 1hw31
    srcs2ref_idx_imgs = []
    srcs2ref_idx_cams = []
    for i in range(srcs_depth_f.shape[0]):
        srcs_idx_cam = idx_img2cam(idx_img, srcs_depth_f[i].unsqueeze(0) , srcs_cam_f[i].unsqueeze(0))
        srcs_idx_world = idx_cam2world(srcs_idx_cam, srcs_cam_f[i].unsqueeze(0))
        srcs2ref_idx_cam = idx_world2cam(srcs_idx_world, ref_cam_r[i].unsqueeze(0))
        srcs2ref_idx_img = idx_cam2img(srcs2ref_idx_cam, ref_cam_r[i].unsqueeze(0))
        srcs2ref_idx_imgs.append(srcs2ref_idx_img)
        srcs2ref_idx_cams.append(srcs2ref_idx_cam)
    srcs2ref_idx_imgs = torch.cat(srcs2ref_idx_imgs, 0)
    srcs2ref_idx_cams = torch.cat(srcs2ref_idx_cams, 0)
    del srcs_idx_cam, srcs_idx_world, srcs2ref_idx_cam, srcs2ref_idx_img
    srcs2ref_xydv = torch.cat(
        [srcs2ref_idx_imgs[..., :2, 0], srcs2ref_idx_cams[..., 2:3, 0], srcs_valid_f.permute(0, 2, 3, 1)],
        dim=-1).permute(0, 3, 1, 2)  # N4hw
    del srcs2ref_idx_imgs, srcs2ref_idx_cams

    reproj_xydv_f, in_range_f = project_img(srcs2ref_xydv, ref_depth_r, srcs_cam_f, ref_cam_r)  # N4hw, N1hw
    reproj_xyd = reproj_xydv_f.view(n, v, 4, h, w)[:, :, :3]
    in_range = (in_range_f * reproj_xydv_f[:, 3:]).view(n, v, 1, h, w)
    return reproj_xyd, in_range

def vis_filter(ref_depth, reproj_xyd, in_range):
    n, v, _, h, w = reproj_xyd.size()
    xy = get_pixel_grids(h, w).permute(3,2,0,1).unsqueeze(1)[:,:,:2]  # 112hw
    dist_masks = (reproj_xyd[:, :, :2, :, :] - xy).norm(dim=2, keepdim=True)  # nv1hw
    depth_masks = (ref_depth.unsqueeze(1) - reproj_xyd[:, :, 2:, :, :]).abs()  # nv1hw
    masks = []
    for i in range(2, v+1):
        geo_mask = in_range * (dist_masks < (i/args.dist_base)).to(ref_depth.dtype) * (depth_masks < (torch.max(ref_depth.unsqueeze(1), reproj_xyd[:, :, 2:, :, :]) * (i/args.rel_diff_base))).to(ref_depth.dtype) # nv1hw
        masks.append(geo_mask.sum(dim=1)) # the final geo_mask needs to be returned for ave_fusion

    geo_mask_sum = geo_mask.sum(dim=1)
    mask = geo_mask_sum >= args.vthresh
    for i in range(2, v+1):
        mask = mask | (masks[i - 2] >= i)

    return geo_mask, mask

def ave_fusion(ref_depth, reproj_xyd, masks):
    ave = ((reproj_xyd[:,:,2:,:,:]*masks).sum(dim=1)+ref_depth) / (masks.sum(dim=1)+1)  # n1hw
    return ave

def med_fusion(ref_depth, reproj_xyd, masks, mask):
    all_d = torch.cat([reproj_xyd[:,:,2:,:,:]*masks, ref_depth.unsqueeze(1)], dim=1)  # n(v+1)1hw
    valid_num = masks.sum(dim=1, keepdim=True) + 1  # n11hw
    gather_idx = (valid_num // 2).long()  # n11hw
    med = all_d.sort(dim=1, descending=True)[0].gather(dim=1, index=gather_idx).squeeze(1)  # n1hw
    return med * mask

def vis_fusion(pair_folder, scan_folder, out_folder, plyfilename):
    # the pair file
    pair_file = os.path.join(pair_folder, "pair.txt")

    pair = load_pair(pair_file)
    pthresh = [float(v) for v in args.pthresh.split(',')]

    # for each reference view and the corresponding source views
    print('load data -------')
    views = {}
    for id in pair['id_list_all']:
        # load the camera parameters
        intrinsics, extrinsics = read_camera_parameters(
            os.path.join(scan_folder, 'cams/{:0>8}_cam.txt'.format(id)))
        proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
        proj_mat[0, :4, :4] = extrinsics
        proj_mat[1, :3, :3] = intrinsics
        # load the  image
        img = cv2.imread(os.path.join(scan_folder, 'images/{:0>8}.jpg'.format(id))).transpose(2,0,1)[::-1]
        # load the estimated depth
        depth_est = read_pfm(os.path.join(out_folder, 'depth_est/{:0>8}.pfm'.format(id)))[0]
        depth_est = np.expand_dims(depth_est,axis=0)
        # load the photometric mask
        prob = []
        prob.append(read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_0.pfm'.format(id)))[0])
        prob.append(read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_1.pfm'.format(id)))[0])
        prob.append(read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_2.pfm'.format(id)))[0])
        prob.append(read_pfm(os.path.join(out_folder, 'confidence/{:0>8}_3.pfm'.format(id)))[0])
        prob.append(read_pfm(os.path.join(out_folder, 'confidence/{:0>8}.pfm'.format(id)))[0])
        prob = np.stack(prob, axis=0)
        views[id] = {
            'image': img,  # 13hw (after next step)
            'cam': proj_mat,  # 1244
            'depth': depth_est,  # 11hw
            'prob': prob,  # 13hw
        }
        recursive_apply(views[id], lambda arr: torch.from_numpy(np.ascontiguousarray(arr)).float().unsqueeze(0))

    print('prob filter -------')
    for id in pair['id_list']:
        views[id]['mask'] = prob_filter(views[id]['prob'].cuda(), pthresh).cpu()  # 11hw bool
        views[id]['depth'] *= views[id]['mask']

    update = {}
    print('vis filter and ave fusion -------')
    for id in pair['id_list']:
        srcs_id = pair[id]['pair']
        ref_depth_g, ref_cam_g = views[id]['depth'].cuda(), views[id]['cam'].cuda()
        srcs_depth_g, srcs_cam_g = [torch.stack([views[loop_id][attr] for loop_id in srcs_id], dim=1).cuda() for attr in
                                    ['depth', 'cam']]

        reproj_xyd_g, in_range_g = get_reproj(ref_depth_g, srcs_depth_g, ref_cam_g, srcs_cam_g)
        vis_masks_g, vis_mask_g = vis_filter(ref_depth_g, reproj_xyd_g, in_range_g)

        ref_depth_ave_g = ave_fusion(ref_depth_g, reproj_xyd_g, vis_masks_g)

        update[id] = {
            'depth': ref_depth_ave_g.cpu(),
            'mask': vis_mask_g.cpu()
        }
        del ref_depth_g, ref_cam_g, srcs_depth_g, srcs_cam_g, reproj_xyd_g, in_range_g, vis_masks_g, vis_mask_g, ref_depth_ave_g
    for i, id in enumerate(pair['id_list']):
        views[id]['mask'] = views[id]['mask'] & update[id]['mask']
        views[id]['depth'] = update[id]['depth'] * views[id]['mask']

    if args.write_mask:
        print('write masks -------')
        for id in pair['id_list']:
            os.makedirs(f'{scan_folder}/mask/', exist_ok=True)
            cv2.imwrite(f'{scan_folder}/mask/{id.zfill(8)}_mask.png', views[id]['mask'][0,0].numpy().astype(np.uint8)*255)


    pcds = {}
    print('back proj -------')
    for id in pair['id_list']:
        ref_depth_g, ref_cam_g = views[id]['depth'].cuda(), views[id]['cam'].cuda()

        idx_img_g = get_pixel_grids(*ref_depth_g.size()[-2:]).unsqueeze(0)
        idx_cam_g = idx_img2cam(idx_img_g, ref_depth_g, ref_cam_g)
        points_g = idx_cam2world(idx_cam_g, ref_cam_g)[..., :3, 0]  # nhw3
        cam_center_g = (- ref_cam_g[:, 0, :3, :3].transpose(-2, -1) @ ref_cam_g[:, 0, :3, 3:])[..., 0]  # n3
        dir_vec_g = cam_center_g.reshape(-1, 1, 1, 3) - points_g  # nhw3

        p_f = points_g.cpu()[views[id]['mask'].squeeze(1)]  # m3
        c_f = views[id]['image'].permute(0, 2, 3, 1)[views[id]['mask'].squeeze(1)] / 255  # m3
        d_f = dir_vec_g.cpu()[views[id]['mask'].squeeze(1)]  # m3

        pcds[id] = {
            'points': p_f,
            'colors': c_f,
            'dirs': d_f,
        }
        del views[id]

    print('Construct combined PCD')
    all_points, all_colors, all_dirs = \
        [torch.cat([pcds[id][attr] for id in pair['id_list']], dim=0) for attr in ['points', 'colors', 'dirs']]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_points.numpy())
    pcd.colors = o3d.utility.Vector3dVector(all_colors.numpy())

    print('Estimate normal')
    pcd.estimate_normals()
    all_normals_np = np.asarray(pcd.normals)
    is_same_dir = (all_normals_np * all_dirs.numpy()).sum(-1, keepdims=True) > 0
    all_normals_np *= is_same_dir.astype(np.float32) * 2 - 1
    pcd.normals = o3d.utility.Vector3dVector(all_normals_np)

    if args.downsample is not None:
        print('Down sample')
        pcd = pcd.voxel_down_sample(args.downsample)

    o3d.io.write_point_cloud(plyfilename, pcd, print_progress=True)


if __name__ == '__main__':

    testlist = [args.testlist]

    #filter saved depth maps with photometric confidence maps and geometric constraints
    for scan in testlist:
        if args.testlist[:4] == "scan":
            scan_id = int(scan[4:])
            save_name = 'epnet{:0>3}_l3.ply'.format(scan_id)
        else:
            save_name = '{}.ply'.format(scan)
        print(save_name[:-4])
        pair_folder = os.path.join(args.testpath, scan)
        scan_folder = os.path.join(args.outdir, scan)
        out_folder = os.path.join(args.outdir, scan)
        vis_fusion(pair_folder, scan_folder, out_folder, os.path.join(args.outdir, save_name))
