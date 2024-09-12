import numpy as np
import re
import sys


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def save_pfm(filename, image, scale=1):
    file = open(filename, "wb")
    color = None

    image = np.flipud(image)

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode('utf-8') if color else 'Pf\n'.encode('utf-8'))
    file.write('{} {}\n'.format(image.shape[1], image.shape[0]).encode('utf-8'))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write(('%f\n' % scale).encode('utf-8'))

    image.tofile(file)
    file.close()

import random, cv2
class RandomCrop(object):
    def __init__(self, CropSize=0.1):
        self.CropSize = CropSize

    def __call__(self, image, normal):
        h, w = normal.shape[:2]
        img_h, img_w = image.shape[:2]
        CropSize_w, CropSize_h = max(1, int(w * self.CropSize)), max(1, int(h * self.CropSize))
        x1, y1 = random.randint(0, CropSize_w), random.randint(0, CropSize_h)
        x2, y2 = random.randint(w - CropSize_w, w), random.randint(h - CropSize_h, h)

        normal_crop = normal[y1:y2, x1:x2]
        normal_resize = cv2.resize(normal_crop, (w, h), interpolation=cv2.INTER_NEAREST)

        image_crop = image[4*y1:4*y2, 4*x1:4*x2]
        image_resize = cv2.resize(image_crop, (img_w, img_h), interpolation=cv2.INTER_LINEAR)

        # import matplotlib.pyplot as plt
        # plt.subplot(2, 3, 1)
        # plt.imshow(image)
        # plt.subplot(2, 3, 2)
        # plt.imshow(image_crop)
        # plt.subplot(2, 3, 3)
        # plt.imshow(image_resize)
        #
        # plt.subplot(2, 3, 4)
        # plt.imshow((normal + 1.0) / 2, cmap="rainbow")
        # plt.subplot(2, 3, 5)
        # plt.imshow((normal_crop + 1.0) / 2, cmap="rainbow")
        # plt.subplot(2, 3, 6)
        # plt.imshow((normal_resize + 1.0) / 2, cmap="rainbow")
        # plt.show()
        # plt.pause(1)
        # plt.close()

        return image_resize, normal_resize

# def center_image(img):
#     """ normalize image input """
#     img = img.astype(np.float32)
#     var = np.var(img, axis=(0,1), keepdims=True)
#     mean = np.mean(img, axis=(0,1), keepdims=True)
#     return (img - mean) / (np.sqrt(var) + 0.00000001)

def image_net_center(img):
    stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    img = img.astype(np.float32)
    img /= 255.
    std = np.array([[stats['std'][::-1]]], dtype=np.float32)  # RGB to BGR
    mean = np.array([[stats['mean'][::-1]]], dtype=np.float32)
    return (img - mean) / (std + 0.00000001)


def image_net_center_inv(img):
    stats = {'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
    std = np.array([[stats['std'][::-1]]], dtype=np.float32)  # RGB to BGR
    mean = np.array([[stats['mean'][::-1]]], dtype=np.float32)
    return ((img * std + mean)*255).astype(np.uint8)

def random_brightness(img: np.ndarray, max_abs_change=50):
    dv = np.random.randint(-max_abs_change, max_abs_change)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = v.astype(np.int32)
    v = np.clip(v + dv, 0, 255).astype(np.uint8)
    hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_contrast(img: np.ndarray, strength_range=[0.3, 1.5]):
    lo, hi = strength_range
    strength = np.random.rand() * (hi - lo) + lo
    img = img.astype(np.int32)
    img = (img - 128) * strength + 128
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def motion_blur(img: np.ndarray, max_kernel_size=3):
        # Either vertial, hozirontal or diagonal blur
        mode = np.random.choice(['h', 'v', 'diag_down', 'diag_up'])
        ksize = np.random.randint(0, (max_kernel_size+1)/2)*2 + 1  # make sure is odd
        center = int((ksize-1)/2)
        kernel = np.zeros((ksize, ksize))
        if mode == 'h':
            kernel[center, :] = 1.
        elif mode == 'v':
            kernel[:, center] = 1.
        elif mode == 'diag_down':
            kernel = np.eye(ksize)
        elif mode == 'diag_up':
            kernel = np.flip(np.eye(ksize), 0)
        var = ksize * ksize / 16.
        grid = np.repeat(np.arange(ksize)[:, np.newaxis], ksize, axis=-1)
        gaussian = np.exp(-(np.square(grid-center)+np.square(grid.T-center))/(2.*var))
        kernel *= gaussian
        kernel /= np.sum(kernel)
        img = cv2.filter2D(img, -1, kernel)
        return img