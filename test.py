import argparse, os, time, sys, gc, cv2
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
from datasets import find_dataset_def
from models import *
from utils import *
from datasets.data_io import save_pfm, image_net_center_inv
from PIL import Image
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='Predict depth')
parser.add_argument('--model', default='mvsnet', help='select model')
parser.add_argument('--dataset', default='general_eval', help='select dataset')
parser.add_argument('--testpath', default='/mnt/B/MVS_GT/dtu/',help='testing data dir for some scenes')
parser.add_argument('--testpath_single_scene', help='testing data path for single scene')
parser.add_argument('--testlist', default='lists/dtu/test.txt', help='testing scene list')
parser.add_argument('--gpu_device', type=str, default='2', help='gpu no.')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')

parser.add_argument('--loadckpt', default='./checkpoints/model_000015.ckpt', help='load a specific checkpoint')
parser.add_argument('--outdir', default='./outputs', help='output dir')


parser.add_argument('--ndepths', type=str, default="64,32,16", help='ndepths')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--num_groups', type=str, default="8,8,8,8,4", help='num_groups')

parser.add_argument('--interval_scale', type=float, default=1.06, help='the depth interval scale')
parser.add_argument('--num_views', type=int, default=5, help='num of view')

parser.add_argument('--num_worker', type=int, default=4, help='depth_filer worker')
parser.add_argument('--save_freq', type=int, default=20, help='save freq of local pcd')

parser.add_argument('--resize_wh', type=str, default='')
parser.add_argument('--crop_wh', type=str, default='')

# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)

cudnn.benchmark = True
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_device
num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
is_distributed = num_gpus > 1

if args.testpath_single_scene:
    args.testpath = os.path.dirname(args.testpath_single_scene)

num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])


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


# read an image
def read_img(filename):
    img = Image.open(filename)
    # scale 0~255 to 0~1
    np_img = np.array(img, dtype=np.float32) / 255.
    return np_img


# read a binary mask
def read_mask(filename):
    return read_img(filename) > 0.5


# save a binary mask
def save_mask(filename, mask):
    assert mask.dtype == np.bool
    mask = mask.astype(np.uint8) * 255
    Image.fromarray(mask).save(filename)


# read a pair file, [(ref_view1, [src_view1-1, ...]), (ref_view2, [src_view2-1, ...]), ...]
def read_pair_file(filename):
    data = []
    with open(filename) as f:
        num_viewpoint = int(f.readline())
        # 49 viewpoints
        for view_idx in range(num_viewpoint):
            ref_view = int(f.readline().rstrip())
            src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
            if len(src_views) > 0:
                data.append((ref_view, src_views))
    return data

def write_cam(file, cam):
    f = open(file, "w")
    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(cam[0][i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(cam[1][i][j]) + ' ')
        f.write('\n')

    f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()

def save_depth(testlist):

    for scene in testlist:
        save_scene_depth([scene])

# run CasMVS model to save depth maps and confidence maps
def save_scene_depth(testlist):
    # dataset, dataloader
    MVSDataset = find_dataset_def(args.dataset)
    test_dataset = MVSDataset(args.testpath, testlist, "test", args.num_views, args.resize_wh, args.crop_wh, args.numdepth, args.interval_scale,)
    TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=4, drop_last=False)

    # model
    model = MVSNet(ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                   depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                   num_groups=[int(ng) for ng in args.num_groups.split(",") if ng])

    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict['model'], strict=True)
    model = nn.DataParallel(model)
    model.cuda()
    model.eval()

    with torch.no_grad():
        for batch_idx, sample in enumerate(TestImgLoader):
            torch.cuda.reset_peak_memory_stats()
            sample_cuda = tocuda(sample)
            start_time = time.time()
            outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_min"], sample_cuda["depth_max"], sample_cuda["depth_interval"])
            end_time = time.time()
            outputs = tensor2numpy(outputs)
            del sample_cuda
            filenames = sample["filename"]
            cams = sample["proj_matrices"]["stage{}".format(str(num_stage-2))].numpy()
            imgs = sample["imgs"].numpy()
            print('Iter {}/{}, Time:{}, Res:{}'.format(batch_idx, len(TestImgLoader), end_time - start_time, imgs[0].shape))

            # save depth maps and confidence maps

            img = imgs[0][0]  #ref view
            cam = cams[0][0]  #ref cam
            depth_filename = os.path.join(args.outdir, filenames[0].format('depth_est', '.pfm'))
            confidence_filename = os.path.join(args.outdir, filenames[0].format('confidence', '.pfm'))
            # uncert_filename = os.path.join(args.outdir, filenames[0].format('uncert', '.pfm'))
            cam_filename = os.path.join(args.outdir, filenames[0].format('cams', '_cam.txt'))
            img_filename = os.path.join(args.outdir, filenames[0].format('images', '.jpg'))
            os.makedirs(depth_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(confidence_filename.rsplit('/', 1)[0], exist_ok=True)
            # os.makedirs(uncert_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(cam_filename.rsplit('/', 1)[0], exist_ok=True)
            os.makedirs(img_filename.rsplit('/', 1)[0], exist_ok=True)
            #save depth maps
            save_pfm(depth_filename, outputs["depth"][0])
            #save confidence maps
            save_pfm(confidence_filename, outputs["photometric_confidence"][0])
            #save cams, img
            write_cam(cam_filename, cam)

            img = np.transpose(img, (1, 2, 0))
            img = image_net_center_inv(img)
            img = cv2.resize(img, (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5)), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(img_filename, img)

            confidence_filename = os.path.join(args.outdir, filenames[0].format('confidence', '_0.pfm'))
            save_pfm(confidence_filename, outputs["stage1"]["photometric_confidence"][0])
            confidence_filename = os.path.join(args.outdir, filenames[0].format('confidence', '_1.pfm'))
            save_pfm(confidence_filename, outputs["stage2"]["photometric_confidence"][0])
            confidence_filename = os.path.join(args.outdir, filenames[0].format('confidence', '_2.pfm'))
            save_pfm(confidence_filename, outputs["stage3"]["photometric_confidence"][0])
            confidence_filename = os.path.join(args.outdir, filenames[0].format('confidence', '_3.pfm'))
            save_pfm(confidence_filename, outputs["stage4"]["photometric_confidence"][0])


    torch.cuda.empty_cache()
    gc.collect()



if __name__ == '__main__':

    print(args.testlist)
    save_scene_depth([args.testlist])

