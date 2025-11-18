import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
import cv2
import torchvision.utils as vutils
import numpy as np

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# NOTE: test parameter setting
KEY_INPUT = 3  # input channels
KEY_OUTPUT = 1
NET_LAYERS = 8  # network depth
NET_CHANNELS = 24  # network width

descriptor_choice = 'swd'  # swd , vgg , acf
temp_size = 7  # swd temp
stride = 1  # swd stride

"""controllable generation parameter"""
phi = 0.05
th = 0.05
min_phi = -0.05

debug_dir = "test"  # debug file
random_seed = 0
# random.seed(random_seed)
# torch.manual_seed(random_seed)


def read_bmp_sequence(image_folder):
    images = []
    for filename in sorted(os.listdir(image_folder)):
        if filename.endswith(".bmp"):
            img = cv2.imread(os.path.join(image_folder, filename), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)

    np_img = np.array(images)
    np_img = np.transpose(np_img, (1, 2, 0))
    np_img = np_img / 255.
    image_mask = torch.from_numpy(np_img)
    return image_mask


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


# NOTE: network design
lmcn = SWNN3D(net_chin=KEY_INPUT, net_chout=KEY_OUTPUT, net_layers=NET_LAYERS, net_channels=NET_CHANNELS)
lmcn.load_state_dict(torch.load(debug_dir + '/params.pytorch'))
lmcn.cuda()
lmcn.eval()

# NOTE: controllable generation
time_begin = time.time()
pyd_phi = [phi]
key = 0
if key < 1:
    while True:
        phi = phi - th
        if phi < min_phi:
            break
        pyd_phi = [phi] + pyd_phi
idx_len = len(pyd_phi)
pyd_idx = [idx_len]
while True:
    idx_len = idx_len - 1
    if idx_len < 1:
        break
    pyd_idx = [idx_len] + pyd_idx

rec_size = 128
with torch.no_grad():
    for idx in range(len(pyd_phi)):
        tmp = torch.normal(0, 1, size=(
            1, KEY_INPUT, rec_size + 2 * NET_LAYERS, rec_size + 2 * NET_LAYERS, rec_size + 2 * NET_LAYERS))
        tmp = tmp + pyd_phi[idx]
        area_in = Variable(tmp, volatile=True).cuda()
        area_out = lmcn(area_in)
        for i in range(0, rec_size, 1):
            slice = area_out[:, :, :, :, i]
            save_image(slice, os.path.join(debug_dir, f'rec/rec_gray/{idx}', f'gray_result-slice-{i}.png'))
            slice[slice >= 0] = 1
            slice[slice < 0] = -1
            save_image(slice, os.path.join(debug_dir, f'rec/rec_bry/{idx}', f'bry_result-slice-{i}.png'))
        del tmp, area_in, area_out

# NOTE: parameter setting add_mean_1
rec_H = 128
rec_W = 128
rec_L = 128
add_mean_1 = torch.zeros(1, 3, rec_H + 2 * NET_LAYERS, rec_W + 2 * NET_LAYERS, rec_L + 2 * NET_LAYERS)
phi_a = 0.00
phi_b = 0.50
l_a = 80
l_b = 160
for i in range(0, rec_H + 2 * NET_LAYERS, 1):
    if i < l_a:
        add_mean_1[:, :, i, :, :] = phi_a
    elif i < l_b:
        add_mean_1[:, :, i, :, :] = phi_a + 1.0 * (i - l_a) * (phi_b - phi_a) / (l_b - l_a)
    else:
        add_mean_1[:, :, i, :, :] = phi_b
for i in range(0, rec_L, 1):
    slice = add_mean_1[:, :, :, :, i]
    save_image(slice, os.path.join(debug_dir, f'add_mean_1', f'add_mean_1_layer{i}.png'))

with torch.no_grad():
    for idx in range(5):
        z_image = torch.normal(0, 1, size=(
            1, KEY_INPUT, rec_H + 2 * NET_LAYERS, rec_W + 2 * NET_LAYERS, rec_L + 2 * NET_LAYERS))
        z_image = z_image + add_mean_1
        z_samples = Variable(z_image, volatile=True).cuda()
        rec_sample = lmcn(z_samples)
        for i in range(0, rec_L, 1):
            slice = rec_sample[:, :, :, :, i]
            save_image(slice, os.path.join(debug_dir, f'add_mean_1/rec/rec_gray/{idx}', f'gray_result-slice-{i}.png'))
            slice[slice >= 0] = 1
            slice[slice < 0] = -1
            save_image(slice, os.path.join(debug_dir, f'add_mean_1/rec/rec_bry/{idx}', f'bry_result-slice-{i}.png'))

# NOTE: parameter setting add_mean_2
add_mean_2 = torch.zeros(1, 3, rec_H + 2 * NET_LAYERS, rec_W + 2 * NET_LAYERS, rec_L + 2 * NET_LAYERS)
phi_c = 0.50
phi_d = -0.00
l_c = int(rec_H + rec_W / 2)
l_d = int(rec_H + rec_W / 2 + rec_W / 4)
for i in range(0, rec_H + 2 * NET_LAYERS, 1):
    for j in range(0, rec_W + 2 * NET_LAYERS, 1):
        if (i + j) < l_c:
            add_mean_2[:, :, i, j, :] = phi_d
        elif (i + j) < l_d:
            add_mean_2[:, :, i, j, :] = phi_c
        else:
            add_mean_2[:, :, i, j, :] = phi_d
for i in range(0, rec_L, 1):
    slice = add_mean_2[:, :, :, :, i]
    save_image(slice, os.path.join(debug_dir, f'add_mean_2', f'add_mean_2_layer{i}.png'))

with torch.no_grad():
    for idx in range(5):
        z_image = torch.normal(0, 1, size=(
            1, KEY_INPUT, rec_H + 2 * NET_LAYERS, rec_W + 2 * NET_LAYERS, rec_L + 2 * NET_LAYERS))
        z_image = z_image + add_mean_2
        z_samples = Variable(z_image, volatile=True).cuda()
        rec_sample = lmcn(z_samples)
        for i in range(0, rec_L, 1):
            slice = rec_sample[:, :, :, :, i]
            save_image(slice, os.path.join(debug_dir, f'add_mean_2/rec/rec_gray/{idx}', f'gray_result-slice-{i}.png'))
            slice[slice >= 0] = 1
            slice[slice < 0] = -1
            save_image(slice, os.path.join(debug_dir, f'add_mean_2/rec/rec_bry/{idx}', f'bry_result-slice-{i}.png'))

time_end = time.time()
time = time_end - time_begin
print(time)
txt_path = './' + debug_dir + '/rec_time.txt'
file_handle = open(txt_path, mode='w')
file_handle.write("time:" + str(time) + '\n')
