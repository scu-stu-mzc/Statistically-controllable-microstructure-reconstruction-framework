import os
import random
import datetime
import numpy
import math
import time
from PIL import Image
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms

import sys
import cv2
import torchvision.utils as vutils
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

# NOTE: test parameter setting, but network parameter should be the same with training parameter
KEY_INPUT = 3  # input channels
KEY_OUTPUT = 1
NET_LAYERS = 8  # network depth
NET_CHANNELS = 24  # network width

# reconstruction size
rec_H = 2048
rec_W = 2048
rec_L = 2048
piece_h = 128
piece_w = 128
piece_l = 128

descriptor_choice = 'swd'  # swd , vgg , acf
temp_size = 7  # swd temp
stride = 1  # swd stride

debug_dir = "test"  # debug file
training_img_path = "training_images/M1.bmp"  # training image path
random_seed = 0
# random.seed(random_seed)
# torch.manual_seed(random_seed)


def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vutils.save_image(torch.clip(img, -1, 1), path, normalize=True)


def cv2ptgray(img):
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    return img


def cv2ptcolor(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float64) / 255.
    img = img * 2 - 1
    img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    return img


def extract_patterns_txt(x, temp_size, stride):
    b, c, h, w = x.shape
    unfold = torch.nn.Unfold(kernel_size=temp_size, stride=stride)
    x_patches = unfold(x).transpose(1, 2).reshape(b, -1, 1, temp_size, temp_size)
    return x_patches.view(-1, b, temp_size, temp_size)


def segment(img, target3d):
    thold = 0.0
    target_parm = torch.mean(img)
    min_err = 2.0
    for tmp_i in range(64, 192, 1):
        tmp_thold = (tmp_i / 255.0) * 2 - 1
        tmp_img3d = target3d.clone()
        tmp_img3d[tmp_img3d >= tmp_thold] = 1
        tmp_img3d[tmp_img3d < tmp_thold] = -1
        tmp_err = torch.abs(torch.mean(tmp_img3d) - target_parm)
        if tmp_err <= min_err:
            min_err = tmp_err
            thold = tmp_thold
    print("parm:", target_parm)
    print("min error:", min_err)
    print("thold:", thold)
    return thold


# NOTE: network structure
class Conv3_3dBlock(nn.Module):
    def __init__(self, input_channels, output_channels, m=0.1):
        super(Conv3_3dBlock, self).__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, 3, padding=0, bias=True)
        self.bn = nn.BatchNorm3d(output_channels, momentum=m)

    def forward(self, x):
        x = F.leaky_relu(self.bn(self.conv(x)))
        return x


class Conv3_3dBlock_act(nn.Module):
    def __init__(self, input_channels, output_channels, m=0.1):
        super(Conv3_3dBlock_act, self).__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, 3, padding=0, bias=True)

    def forward(self, x):
        x = F.leaky_relu(self.conv(x))  # sometimes work
        return x


class SWNN3D(nn.Module):
    def __init__(self, net_chin=3, net_chout=3, net_layers=8, net_channels=16):
        super(SWNN3D, self).__init__()
        self.convs = []
        conv_layer_1 = Conv3_3dBlock_act(net_chin, net_channels)
        setattr(self, 'cb1_1', conv_layer_1)
        self.convs.append(conv_layer_1)
        for i in range(net_layers - 1):
            conv_layer_n = Conv3_3dBlock(net_channels, net_channels)
            setattr(self, 'cb%i_1' % (i + 2), conv_layer_n)
            self.convs.append(conv_layer_n)
        last_conv = nn.Conv3d(net_channels, net_chout, 1, padding=0, bias=True)
        setattr(self, 'last_conv', last_conv)
        self.convs.append(last_conv)

    def forward(self, z):
        y = z
        for i in range(NET_LAYERS + 1):
            y = self.convs[i](y)
        return y


# training image read
training_img = cv2.imread(training_img_path, 0)
if len(training_img.shape) == 2:
    print(training_img.shape)  # h w
    training_img = cv2ptgray(training_img)  # c,h,w
    if KEY_OUTPUT == 3:
        training_img = torch.cat([training_img] * 3, dim=0)
    training_img = training_img.unsqueeze(0).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # b c h w
else:
    print(training_img.shape)  # h w c
    training_img = cv2ptcolor(training_img)  # c,h,w
    training_img = training_img.unsqueeze(0).to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    # b c h w

# NOTE: network design
lmcn = SWNN3D(net_chin=KEY_INPUT, net_chout=KEY_OUTPUT, net_layers=NET_LAYERS, net_channels=NET_CHANNELS)
lmcn.load_state_dict(torch.load(debug_dir + '/params.pytorch'))
lmcn.cuda()
lmcn.eval()

# base
time_begin = time.time()
with torch.no_grad():
    for idx in range(5):
        z_image = torch.normal(0, 1, size=(
            1, KEY_INPUT, rec_H + 2 * NET_LAYERS, rec_W + 2 * NET_LAYERS, rec_L + 2 * NET_LAYERS))
        rec_out = torch.zeros(1, 1, rec_H, rec_W, rec_L)
        num_block = 0
        for i in range(0, rec_H, piece_h):
            for j in range(0, rec_W, piece_w):
                for k in range(0, rec_L, piece_l):
                    tmp = z_image[:, :, i:i + piece_h + 2 * NET_LAYERS, j:j + piece_w + 2 * NET_LAYERS,
                          k:k + piece_l + 2 * NET_LAYERS]
                    area_in = Variable(tmp, volatile=True).cuda()
                    area_out = lmcn(area_in)
                    rec_out[:, :, i:i + piece_h, j:j + piece_w, k:k + piece_l] = area_out
                    num_block = num_block + 1
                    print(num_block)
                    del tmp, area_in, area_out

        """if target training image is read correctly, use 'segment_value' function for segment"""
        # thold = segment(training_img, rec_sample)
        thold = 0.0
        if rec_L > 1024:
            if idx == 0:
                for i in range(0, rec_L, 1):
                    slice = rec_out[:, :, :, :, i]
                    # save_image(slice, os.path.join(debug_dir, f'rec/rec_gray/{idx}', f'gray_result-slice-{i:04}.bmp'))
                    slice[slice >= thold] = 1
                    slice[slice < thold] = -1
                    save_image(slice, os.path.join(debug_dir, f'rec/rec_bry/{idx}', f'bry_result-slice-{i:04}.jpg'))
        else:
            for i in range(0, rec_L, 1):
                slice = rec_out[:, :, :, :, i]
                # save_image(slice, os.path.join(debug_dir, f'rec/rec_gray/{idx}', f'gray_result-slice-{i:04}.bmp'))
                slice[slice >= thold] = 1
                slice[slice < thold] = -1
                save_image(slice, os.path.join(debug_dir, f'rec/rec_bry/{idx}', f'bry_result-slice-{i:04}.jpg'))


time_end = time.time()
time = time_end - time_begin
print(time)
txt_path = './' + debug_dir + '/rec_time.txt'
file_handle = open(txt_path, mode='w')
file_handle.write("time:" + str(time) + '\n')
file_handle.write("size:" + str(rec_H) + '*' + str(rec_W) + '*' + str(rec_L) + '\n')
