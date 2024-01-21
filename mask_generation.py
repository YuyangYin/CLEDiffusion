import os
import os.path as osp
import numpy as np
import numpy.random as npr
import torch
import torchvision.transforms as tvtrans
import PIL.Image
from PIL import Image, ImageDraw
import math
import json
PIL.Image.MAX_IMAGE_PIXELS = None
import cv2
# from .common import *

# from ..log_service import print_log

import numpy.random as npr
########################
# RandomBrush for mask #
########################

def RandomBrush(
    max_tries,
    s,
    min_num_vertex = 4,
    max_num_vertex = 18,
    mean_angle = 2*math.pi / 5,
    angle_range = 2*math.pi / 15,
    min_width = 12,
    max_width = 48):
    H, W = s, s
    average_radius = math.sqrt(H*H+W*W) / 8
    mask = Image.new('L', (W, H), 0)
    for _ in range(np.random.randint(max_tries)):
        num_vertex = np.random.randint(min_num_vertex, max_num_vertex)
        angle_min = mean_angle - np.random.uniform(0, angle_range)
        angle_max = mean_angle + np.random.uniform(0, angle_range)
        angles = []
        vertex = []
        for i in range(num_vertex):
            if i % 2 == 0:
                angles.append(2*math.pi - np.random.uniform(angle_min, angle_max))
            else:
                angles.append(np.random.uniform(angle_min, angle_max))

        h, w = mask.size
        vertex.append((int(np.random.randint(0, w)), int(np.random.randint(0, h))))
        for i in range(num_vertex):
            r = np.clip(
                np.random.normal(loc=average_radius, scale=average_radius//2),
                0, 2*average_radius)
            new_x = np.clip(vertex[-1][0] + r * math.cos(angles[i]), 0, w)
            new_y = np.clip(vertex[-1][1] + r * math.sin(angles[i]), 0, h)
            vertex.append((int(new_x), int(new_y)))

        draw = ImageDraw.Draw(mask)
        width = int(np.random.uniform(min_width, max_width))
        draw.line(vertex, fill=1, width=width)
        for v in vertex:
            draw.ellipse((v[0] - width//2,
                          v[1] - width//2,
                          v[0] + width//2,
                          v[1] + width//2),
                         fill=1)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        if np.random.random() > 0.5:
            mask.transpose(Image.FLIP_TOP_BOTTOM)
    mask = np.asarray(mask, np.uint8)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 0)
    if np.random.random() > 0.5:
        mask = np.flip(mask, 1)
    return mask

def RandomMask(s, hole_range=[0,1]):
    coef = min(hole_range[0] + hole_range[1], 1.0)
    if np.random.random() < 0:
        return torch.ones(1, s, s).float()
    while True:
        mask = np.ones((s, s), np.uint8)
        def Fill(max_size):
            w, h = np.random.randint(max_size), np.random.randint(max_size)
            ww, hh = w // 2, h // 2
            x, y = np.random.randint(-ww, s - w + ww), np.random.randint(-hh, s - h + hh)
            mask[max(y, 0): min(y + h, s), max(x, 0): min(x + w, s)] = 0
        def MultiFill(max_tries, max_size):
            for _ in range(np.random.randint(max_tries)):
                Fill(max_size)
        MultiFill(int(10 * coef), s // 2)
        MultiFill(int(5 * coef), s)
        mask = np.logical_and(mask, 1 - RandomBrush(int(20 * coef), s))
        hole_ratio = 1 - np.mean(mask)
        if hole_range is not None and (hole_ratio <= hole_range[0] or hole_ratio >= hole_range[1]):
            continue
        return torch.from_numpy(mask[np.newaxis, ...].astype(np.float32)).float()




# @regformat()
class RandomMaskFormatter(object):
    """
    This formatter is a direct replication of the original CoModGan TF code
    """
    def __init__(self, random_flip=True, mask_resolution=256, hole_range=[0, 1]):
        self.random_flip = random_flip
        self.mask_resolution = mask_resolution
        self.hole_range = hole_range

    def __call__(self, element):
        x = (element['image']*2 - 1)
        if (self.random_flip) and (npr.rand() < 0.5):
            x = x.flip(-1)
        mask = RandomMask(self.mask_resolution, self.hole_range)[0]
        return x, mask, element['unique_id']

#######################
# LAMA mask generator #
#######################

# @regformat()
class LamaMaskFormatter(object):
    """
    a) The one that do not generate LAMA style mask.
    """
    def __init__(self, random_flip=True, resolution=256, type='thin',):
        from .lama_mask_utils import MixedMaskGenerator
        if type == 'thin' and resolution == 256:
            from .lama_mask_utils import setting_thin256 as setting
        elif type == 'medium' and resolution == 256:
            from .lama_mask_utils import setting_medium256 as setting
        elif type == 'thick' and resolution == 256:
            from .lama_mask_utils import setting_thick256 as setting
        elif type == 'thin' and resolution == 512:
            from .lama_mask_utils import setting_thin512 as setting
        elif type == 'medium' and resolution == 512:
            from .lama_mask_utils import setting_medium512 as setting
        elif type == 'thick' and resolution == 512:
            from .lama_mask_utils import setting_thick512 as setting
        else:
            raise ValueError
        self.mask_maker = MixedMaskGenerator(**setting)
        self.random_flip = random_flip

    def __call__(self, element):
        x = (element['image']*2 - 1)
        if (self.random_flip) and (npr.rand() < 0.5):
            x = x.flip(-1)
        mask = self.mask_maker(x)
        mask = 1-mask[0] # the generate mask is flipped (1-missing, 0-no-missing) -> (0-missing, 1-no-missing)
        return x, mask, element['unique_id']

###############################################
# use for evaluation with all image generated #
###############################################
# a) the dataloader that load generated image directly from loader

# @regformat()
class NoMaskFormatter(object):
    """
    a) The one that do not generate mask.
    b) Direct output the fake result
    """
    def __init__(self):
        pass

    def __call__(self, element):
        x = element['image']
        gen = element['gen']
        return x, gen, element['unique_id']

if __name__ == '__main__':
    for i in range(500):
        res = RandomMask(128, hole_range=[0, 0.6])
        res=cv2.GaussianBlur(res[0].numpy(),(25,25),0)
        save_dir=r'./data/mask/'
        os.makedirs(save_dir,exist_ok=True)
        save_name=save_dir+ str(i)+'.png'
        print(save_name)
        cv2.imwrite(save_name,(res * 255).astype(np.uint8))
