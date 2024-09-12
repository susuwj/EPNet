import argparse, sys
import os
import open3d
from plyfile import PlyData, PlyElement
import numpy as np
from utils import *

parser = argparse.ArgumentParser(description='Transfer the ply files of ETH3D for evalution')
parser.add_argument('--testlist', default='', help='testing scene list')
parser.add_argument('--outdir', default='./outputs', help='output dir')


# parse arguments and check
args = parser.parse_args()
print("argv:", sys.argv[1:])
print_args(args)


def transfer(temp_plyfilename, plyfilename):
    print(temp_plyfilename)
    plydata = PlyData.read(plyfilename)
    vertex = np.array([tuple(np.array([plydata.elements[0].data[i][0].astype(np.float32),plydata.elements[0].data[i][1].astype(np.float32),plydata.elements[0].data[i][2].astype(np.float32),plydata.elements[0].data[i][-3].astype(np.uint8),plydata.elements[0].data[i][-2].astype(np.uint8),plydata.elements[0].data[i][-1].astype(np.uint8)])) for i in range(plydata.elements[0].count)], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    el = PlyElement.describe(vertex, 'vertex')
    PlyData([el]).write(temp_plyfilename)

if __name__ == '__main__':
    testlist = [args.testlist]
    for scan in testlist:
        save_name = '{}.ply'.format(scan)
        print(save_name[:-4])
        os.makedirs(os.path.join(args.outdir, 'temp'), exist_ok=True)
        transfer(os.path.join(args.outdir, 'temp', save_name), os.path.join(args.outdir, save_name))

 
