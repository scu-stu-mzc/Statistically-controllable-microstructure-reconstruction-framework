import os
import random
import numpy
import math
import time
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


# NOTE: training parameter setting
KEY_INPUT = 3  # input channels
KEY_OUTPUT = 1  # output channels
NET_LAYERS = 8  # network depth
NET_CHANNELS = 24  # network width

descriptor_choice = 'swd'  # swd , vgg , acf

temp_size = 7  # swd temp
stride = 1  # swd stride

"""to perform controllable generation, set key_control = 1, otherwise set key_control = 0"""
key_control = 1  # key to controllable generation
key_multi_train = 1  # key to multiple training target parameter  in controllable generation
phi = 0.30  # max given parameter
th = 0.05  # sampling interval, training target parameter: phi, phi-th, ..., phi-n*th
min_phi = 0.15  # cut-off parameter

debug_dir = "midresult"  # debug file
training_img_path = "training_images/M1.bmp"  # training image path
ref_size = 128  # image size
learning_rate = 0.05
iterations = 2000
save_slice = 100
batch_size = 1
random_seed = 0
random.seed(random_seed)
torch.manual_seed(random_seed)

initial_noise = torch.normal(0, 1, size=(
    1, KEY_INPUT, ref_size + 2 * NET_LAYERS, ref_size + 2 * NET_LAYERS, ref_size + 2 * NET_LAYERS))


# NOTE: swd related
def expanding_operation(mp1, mp2):
    if mp1.shape[1] == mp2.shape[1]:
        return mp1, mp2
    elif mp1.shape[1] < mp2.shape[1]:
        t_mp = mp1
        mp1 = mp2
        mp2 = t_mp
    t_times = mp1.shape[1] // mp2.shape[1]
    mp2 = torch.cat([mp2] * t_times, dim=1)
    if mp1.shape[1] > mp2.shape[1]:
        t_mod = torch.randperm(mp2.shape[1])[:mp1.shape[1] - mp2.shape[1]]
        mp2 = torch.cat([mp2, mp2[:, t_mod]], dim=1)
    return mp1, mp2


def expanding_operation_v2(mp1, mp2):
    if mp1.shape[2] == mp2.shape[2]:
        return mp1, mp2
    elif mp1.shape[2] < mp2.shape[2]:
        t_mp = mp1
        mp1 = mp2
        mp2 = t_mp
    t_times = mp1.shape[2] // mp2.shape[2]
    mp2 = torch.cat([mp2] * t_times, dim=2)
    if mp1.shape[2] > mp2.shape[2]:
        t_mod = torch.randperm(mp2.shape[2])[:mp1.shape[2] - mp2.shape[2]]
        mp2 = torch.cat([mp2, mp2[:, t_mod]], dim=2)
    return mp1, mp2


class SingleMPSWDLoss2D(torch.nn.Module):
    def __init__(self, temp_size=7, stride=1, num_proj=256, channels=3):
        super(SingleMPSWDLoss2D, self).__init__()
        self.temp_size = temp_size
        self.stride = stride
        self.num_proj = num_proj
        self.channels = channels

    def forward(self, x, y):
        # b, c, h, w = x.shape
        rand = torch.randn(self.num_proj, self.channels, self.temp_size, self.temp_size).to(x.device)
        if self.num_proj > 1:
            rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize

        projx = F.conv2d(x, rand).reshape(self.num_proj, -1)

        projy = F.conv2d(y, rand).reshape(self.num_proj, -1)

        projx, projy = expanding_operation(projx, projy)

        projx, _ = torch.sort(projx, dim=1)
        projy, _ = torch.sort(projy, dim=1)

        loss = torch.abs(projx - projy).mean()

        return loss


def extract_patterns_txt(x, temp_size, stride):
    b, c, h, w = x.shape
    unfold = torch.nn.Unfold(kernel_size=temp_size, stride=stride)
    x_patches = unfold(x).transpose(1, 2).reshape(b, -1, 1, temp_size, temp_size)
    return x_patches.view(-1, b, temp_size, temp_size)


def sort_with_idx(source, idx):
    """  source: b len 1 t*t,  idx: b len 1 1 """
    # b, len, 1, p = source.size()
    after_sort = torch.zeros_like(source)
    for i in range(idx.shape[1]):
        after_sort[:, i, :, :] = source[:, idx[0, i, 0, 0], :, :]
    return after_sort


def resample_cg(y, p, temp_size, stride):
    """ y: b 1 h w, p: 0-0.5 """
    b, c, h, w = y.shape
    print("y:", y.shape)
    # p = 0.1
    t_p = p * 1 + (1 - p) * (-1)
    avg_y = torch.mean(y)
    print("avg_y:", avg_y)
    unfold = torch.nn.Unfold(kernel_size=temp_size, stride=stride)  # b t*t len
    y_patches = unfold(y).transpose(1, 2).reshape(b, -1, 1, temp_size, temp_size).view(b, -1, 1, temp_size * temp_size)
    # b len 1 t*t
    print("y_patches:", y_patches.shape)
    avg_y_patches = torch.mean(y_patches)
    print("avg_y_patches:", avg_y_patches)
    y_patches_avg = torch.mean(y_patches, dim=3, keepdim=True)  # b len 1 t*t
    print("y_patches_avg:", y_patches_avg.shape)
    y_patches_avg_sort, idx = torch.sort(y_patches_avg, dim=1)  # b len 1 1
    print("y_patches_avg_sort:", y_patches_avg_sort.shape)
    print("idx:", idx.shape)
    target_num = 0
    rest_num = 0
    rest_seq = 0
    # t_p = avg_y  # test
    if t_p < avg_y_patches * (1 + 0.00):
        print("small")
        total_num = y_patches_avg_sort.shape[1]
        for seq1 in range(y_patches_avg_sort.shape[1]):
            tem_sum = torch.mean(y_patches_avg_sort[:, :seq1, :, :])
            if tem_sum >= t_p:
                print("find small seq 1")
                target_num = seq1
                break
        target_seq = y_patches_avg_sort[:, :target_num, :, :]
        total_target_num = target_num
        while (total_target_num + target_num <= total_num):
            print("copy small seq")
            target_seq = torch.cat([target_seq, target_seq], dim=1)
            total_target_num = total_target_num + target_num
        rest_seq = total_num - total_target_num
        for seq2 in range(target_num):
            tem_sum = torch.mean(y_patches_avg_sort[:, seq2:seq2 + rest_seq, :, :])
            if tem_sum >= t_p:
                print("find small seq 2")
                rest_num = seq2
                break
        target_seq = torch.cat([target_seq, y_patches_avg_sort[:, rest_num:rest_num + rest_seq, :, :]], dim=1)
        print(target_seq.shape)
        print(torch.mean(target_seq))
        print("samll seq finish")
        """ create target seq """
        y_patches_sort = sort_with_idx(y_patches, idx)
        y_target_seq = y_patches_sort[:, :target_num, :, :]
        y_patches_sort, y_target_seq = expanding_operation(y_patches_sort, y_target_seq)
        print("y_target_seq.shape", y_target_seq.shape)
        print("y_target_seq_avg", torch.mean(y_target_seq))
        return y_target_seq.view(-1, b, temp_size, temp_size)

    elif t_p > avg_y_patches * (1 - 0.00):
        print("large")
        total_num = y_patches_avg_sort.shape[1]
        for seq1 in range(y_patches_avg_sort.shape[1]):
            tem_sum = torch.mean(y_patches_avg_sort[:, total_num - seq1:, :, :])
            if tem_sum <= t_p:
                print("find large seq 1")
                target_num = seq1
                break
        target_seq = y_patches_avg_sort[:, total_num - target_num:, :, :]
        total_target_num = target_num
        while (total_target_num + target_num <= total_num):
            print("copy large seq")
            target_seq = torch.cat([target_seq, target_seq], dim=1)
            total_target_num = total_target_num + target_num
        rest_seq = total_num - total_target_num
        for seq2 in range(target_num):
            tem_sum = torch.mean(y_patches_avg_sort[:, total_num - seq2 - rest_seq:total_num - seq2, :, :])
            if tem_sum <= t_p:
                print("find large seq 2")
                rest_num = seq2
                break
        target_seq = torch.cat(
            [target_seq, y_patches_avg_sort[:, total_num - rest_num - rest_seq:total_num - rest_num, :, :]], dim=1)
        print(target_seq.shape)
        print(torch.mean(target_seq))
        print("large seq finish")
        """ create target seq """
        y_patches_sort = sort_with_idx(y_patches, idx)
        y_target_seq = y_patches_sort[:, y_patches_sort.shape[1] - target_num:, :, :]
        y_patches_sort, y_target_seq = expanding_operation(y_patches_sort, y_target_seq)
        print("y_target_seq.shape", y_target_seq.shape)
        print("y_target_seq_avg", torch.mean(y_target_seq))
        return y_target_seq.view(-1, b, temp_size, temp_size)
    else:
        print("others")
        """ create target seq """
        y_target_seq = y_patches
        print("y_target_seq.shape", y_target_seq.shape)
        print("y_target_seq_avg", torch.mean(y_target_seq))
        return y_target_seq.view(-1, b, temp_size, temp_size)


def Calc_UV(y):
    len, bc, ht, wt = y.shape
    print("y:", y.shape)
    y_patches = y.clone().reshape(len, bc, ht * wt)
    # b len 1 t*t
    print("y_patches:", y_patches.shape)  # len bc 1 t*t
    y_patches_avg = torch.mean(y_patches, dim=2)  # len bc 1
    y_patches_avg = torch.mean(y_patches_avg, dim=1)  # len 1
    print("y_patches_avg:", y_patches_avg.shape)   # len 1
    U = torch.mean(y_patches_avg, dim=0)  # 1
    V = torch.std(y_patches_avg, dim=0)  # 1
    return U, V


class SingleMPSWDLoss2D_cg_act(torch.nn.Module):
    def __init__(self, temp_size=5, stride=1, num_proj=128, channels=1):
        super(SingleMPSWDLoss2D_cg_act, self).__init__()
        self.temp_size = temp_size
        self.stride = stride
        self.num_proj = num_proj
        self.channels = channels

    def forward(self, x, y):
        b, c, h, w = x.shape
        rand = torch.randn(self.num_proj, self.channels, self.temp_size, self.temp_size).to(x.device)
        if self.num_proj > 1:
            rand = rand / torch.std(rand, dim=0, keepdim=True)  # noramlize

        outx = F.conv2d(x, rand)
        outy = F.conv2d(y, rand).transpose(0, 1)
        projx = outx.reshape(b, self.num_proj, -1)  # b proj hxw
        projy = outy.reshape(1, self.num_proj, -1)

        projx, projy = expanding_operation_v2(projx, projy)
        projx, _ = torch.sort(projx, dim=2)
        projy, _ = torch.sort(projy, dim=2)

        loss = torch.abs(projx - projy).mean()
        return loss


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


def downscale(img, pyr_factor):
    assert 0 < pyr_factor < 1
    c, y, x = img.shape
    new_x = int(x * pyr_factor)
    new_y = int(y * pyr_factor)
    return transforms.Resize((new_y, new_x))(img)


def get_pyramid(img, min_size, pyr_factor):
    pyd = [img]
    while True:
        img = downscale(img, pyr_factor)
        if img.shape[-2] < min_size:
            break
        pyd = [img] + pyd
    return pyd


# NOTE: description function
class ACFloss(nn.Module):
    def __init__(self):
        super(ACFloss, self).__init__()

    def forward(self, x, t):
        p2d = (int(ref_size / 2), int(ref_size / 2), int(ref_size / 2), int(ref_size / 2))
        x_pad = F.pad(x, p2d, 'circular')
        t_pad = F.pad(t, p2d, 'circular')
        x_out1 = F.conv2d(x_pad[:, 0, :, :].unsqueeze(0), x[:, 0, :, :].unsqueeze(0))
        t_out1 = F.conv2d(t_pad[:, 0, :, :].unsqueeze(0), t[:, 0, :, :].unsqueeze(0))
        x_out2 = F.conv2d(x_pad[:, 1, :, :].unsqueeze(0), x[:, 1, :, :].unsqueeze(0))
        t_out2 = F.conv2d(t_pad[:, 1, :, :].unsqueeze(0), t[:, 1, :, :].unsqueeze(0))
        x_out3 = F.conv2d(x_pad[:, 2, :, :].unsqueeze(0), x[:, 2, :, :].unsqueeze(0))
        t_out3 = F.conv2d(t_pad[:, 2, :, :].unsqueeze(0), t[:, 2, :, :].unsqueeze(0))
        loss1 = torch.abs(x_out1 - t_out1).mean() / 100
        loss2 = torch.abs(x_out2 - t_out2).mean() / 100
        loss3 = torch.abs(x_out3 - t_out3).mean() / 100
        loss4 = loss1 + loss2 + loss3
        return loss4


class grayACFloss(nn.Module):
    def __init__(self):
        super(grayACFloss, self).__init__()

    def forward(self, x, t):
        p2d = (int(ref_size / 2), int(ref_size / 2), int(ref_size / 2), int(ref_size / 2))
        # x_pad = F.pad(x, p2d, 'circular')
        # t_pad = F.pad(t, p2d, 'circular')
        x_pad = F.pad(x, p2d, 'constant')
        t_pad = F.pad(t, p2d, 'constant')
        x_out = F.conv2d(x_pad, x)
        t_out = F.conv2d(t_pad, t)
        loss1 = torch.abs(x_out - t_out).mean() / 100
        return loss1


class VGG(nn.Module):
    def __init__(self, pool='max', pad=1):
        super(VGG, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=pad)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=pad)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=pad)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=pad)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=pad)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=pad)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=pad)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=pad)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


class GramMatrix(nn.Module):
    def forward(self, input):
        b, c, h, w = input.size()
        F = input.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        G.div_(h * w * c)
        return G


class GramMSELoss(nn.Module):
    def forward(self, input, target):
        out = nn.MSELoss()(GramMatrix()(input), target)
        return (out)


# NOTE: network structure
class Conv3_3dBlock(nn.Module):
    def __init__(self, input_channels, output_channels, m=0.1):
        super(Conv3_3dBlock, self).__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, 3, padding=0, bias=True)
        self.bn = nn.BatchNorm3d(output_channels, momentum=m)

    def forward(self, x):
        x = F.leaky_relu(self.bn(self.conv(x)))
        return x


class Conv3_3dBlock_nobn(nn.Module):
    def __init__(self, input_channels, output_channels, m=0.1):
        super(Conv3_3dBlock_nobn, self).__init__()
        self.conv = nn.Conv3d(input_channels, output_channels, 3, padding=0, bias=True)

    def forward(self, x):
        x = F.leaky_relu(self.conv(x))
        return x


class SWNN3D(nn.Module):
    def __init__(self, net_chin=3, net_chout=3, net_layers=8, net_channels=16):
        super(SWNN3D, self).__init__()
        self.convs = []
        conv_layer_1 = Conv3_3dBlock_nobn(net_chin, net_channels)
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


def calc_size_input(h, w, d, pad):
    s = [math.ceil(h + 2 * pad),
         math.ceil(w + 2 * pad),
         math.ceil(d + 2 * pad)]
    return s


# NOTE: network design
lmcn = SWNN3D(net_chin=KEY_INPUT, net_chout=KEY_OUTPUT, net_layers=NET_LAYERS, net_channels=NET_CHANNELS)
print(lmcn)  # display network structure
params = list(lmcn.parameters())
total_parameters = 0
for p in params:
    total_parameters = total_parameters + p.data.numpy().size
print('total number of parameters = ' + str(total_parameters))
lmcn.cuda()

# NOTE: description function
if descriptor_choice == 'vgg':
    print("loss choice: vgg loss")
    descriptor = VGG(pool='avg', pad=1)
    descriptor.load_state_dict(torch.load('./Models/vgg_conv.pth'))
    for param in descriptor.parameters():
        param.requires_grad = False
    descriptor = descriptor.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
elif descriptor_choice == 'acf':
    print("loss choice: acf loss")
    if KEY_OUTPUT == 3:
        descriptor = ACFloss()
    else:
        descriptor = grayACFloss()
    descriptor = descriptor.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
else:
    # descriptor = SingleMPSWDLoss2D(temp_size=temp_size, stride=stride, num_proj=256, channels=KEY_OUTPUT)
    descriptor = SingleMPSWDLoss2D_cg_act(temp_size=temp_size, stride=stride, num_proj=512, channels=KEY_OUTPUT)
    descriptor = descriptor.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# NOTE: training image processing
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

if key_control == 1:
    training_patches = resample_cg(training_img, phi, temp_size, stride)
    save_image(training_patches,
               os.path.join(debug_dir, 'optimization', f'test{phi}-{torch.mean(training_patches)}.png'))
    pyd = [training_patches]
    pyd_phi = [torch.mean(training_patches)]
    tmp_U, tmp_V = Calc_UV(training_patches)
    pyd_U = [tmp_U]
    pyd_V = [tmp_V]
    save_image(training_patches,
               os.path.join(debug_dir, 'optimization', f'test{phi}-U-{tmp_U}-V-{tmp_V}.png'))
    key = 0
    if key < 1:
        while True:
            phi = phi - th
            if phi < min_phi:
                break
            target_patches = resample_cg(training_img, phi, temp_size, stride)
            save_image(target_patches,
                       os.path.join(debug_dir, 'optimization', f'test{phi}-{torch.mean(target_patches)}.png'))
            pyd = [target_patches] + pyd
            pyd_phi = [torch.mean(target_patches)] + pyd_phi
            tmp_U, tmp_V = Calc_UV(target_patches)
            pyd_U = [tmp_U] + pyd_U
            pyd_V = [tmp_V] + pyd_V
            save_image(target_patches,
                       os.path.join(debug_dir, 'optimization', f'test{phi}-U-{tmp_U}-V-{tmp_V}.png'))

else:
    target_patches = extract_patterns_txt(training_img, temp_size, stride)
    phi = torch.mean(target_patches)
    save_image(target_patches, os.path.join(debug_dir, 'optimization', f'test{phi}.png'))
    pyd = [target_patches]
    pyd_phi = [phi]

idx_len = len(pyd)
pyd_idx = [idx_len]
while True:
    idx_len = idx_len - 1
    if idx_len < 1:
        break
    pyd_idx = [idx_len] + pyd_idx


# NOTE: Optimization process
# determine the optimizer
optimizer = optim.Adam(lmcn.parameters(), lr=learning_rate)
loss_history = numpy.zeros((iterations, len(pyd)))

# Optimization start
b, _, size_h, size_w = training_img.shape
c = KEY_INPUT
size_l = size_h
save_image(training_img, os.path.join(debug_dir, 'optimization', f'target.png'))
directions = [0, 1, 2]

if descriptor_choice == 'vgg':
    loss_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
    loss_fns = [GramMSELoss()] * len(loss_layers)
    loss_fns = [loss_fn.cuda() for loss_fn in loss_fns]
    w = [1, 1, 1, 1, 1]
    train_images_torch = Variable(training_img)
    target = [GramMatrix()(f).detach() for f in descriptor(train_images_torch, loss_layers)]

time_begin = time.time()
for n_iter in range(iterations):
    optimizer.zero_grad()

    for idd, d in enumerate(directions):
        output_sizes = [ref_size for N in range(3)]
        output_sizes[d] = 1

        for i in range(batch_size):
            # # get input area
            input_size = calc_size_input(output_sizes[0], output_sizes[1], output_sizes[2], NET_LAYERS)

            for idx in range(len(pyd)):
                if key_multi_train == 0:
                    z_image_idx = torch.normal(0, 1,
                                               size=(b, c, input_size[0], input_size[1], input_size[2]))
                else:
                    z_image_idx = torch.normal(pyd_phi[idx], pyd_V[idx],
                                               size=(b, c, input_size[0], input_size[1], input_size[2]))
                z_samples = Variable(
                    z_image_idx.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
                rec_sample = lmcn(z_samples)
                if d == 0:
                    rec_sample = rec_sample[:, :, 0, 0:output_sizes[1]:, 0:output_sizes[2]]

                if d == 1:
                    rec_sample = rec_sample[:, :, 0:output_sizes[0], 0, 0:output_sizes[2]]

                if d == 2:
                    rec_sample = rec_sample[:, :, 0:output_sizes[0], 0:output_sizes[1], 0]


                # NOTE: description function select
                if descriptor_choice == 'vgg':
                    # # vgg
                    rec_domain = descriptor(rec_sample, loss_layers)
                    losses = [w[a] * loss_fns[a](f, target[a]) for a, f in enumerate(rec_domain)]
                    single_loss = (1 / (batch_size * len(pyd_idx) * len(directions))) * losses
                    # # vgg
                elif descriptor_choice == 'acf':
                    # # acf
                    losses = descriptor(rec_sample, pyd[idx])
                    single_loss = (1 / (batch_size * len(pyd_idx) * len(directions))) * losses
                    # # acf
                else:
                    # # swd
                    losses = descriptor(rec_sample, pyd[idx])
                    single_loss = (1 / (batch_size * len(pyd_idx) * len(directions))) * losses
                    # # swd

                single_loss.backward(retain_graph=False)
                loss_history[n_iter, idx] = loss_history[n_iter, idx] + single_loss.item()
                del losses, single_loss
                del z_samples

                if n_iter <= 100:
                    if n_iter % 10 == 0:
                        save_image(rec_sample,
                                   os.path.join(debug_dir, 'optimization',
                                                f'd-{d}-iter-{n_iter + 1}-phi-{len(pyd_phi) - idx}.png'))
                else:
                    if n_iter % save_slice == (save_slice - 1):
                        save_image(rec_sample,
                                   os.path.join(debug_dir, 'optimization',
                                                f'd-{d}-iter-{n_iter + 1}-phi-{len(pyd_phi) - idx}.png'))
                del rec_sample

    print('Iteration: %d, loss: %f' % (n_iter, sum(loss_history[n_iter, :])))
    optimizer.step()

# Optimization end
time_end = time.time()
time = time_end - time_begin
print(time)

z_image_base = initial_noise[:, :, :, :, :]
z_image_base = z_image_base.to(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
for idx in range(len(pyd)):
    z_image = z_image_base
    print(torch.mean(z_image))
    z_samples = Variable(z_image)
    rec_sample = lmcn(z_samples)
    for i in range(0, ref_size, 1):
        # print(i)
        slice = rec_sample[:, :, :, :, i]
        save_image(slice, os.path.join(debug_dir, 'optimization', f'test/phi-{len(pyd_phi) - idx}-slice-{i}.png'))

# save final model and loss history
txt_path = debug_dir + '/optimization/loss.txt'
file_handle = open(txt_path, mode='w')
file_handle.write("time " + str(time) + '\n')
file_handle.write("seed " + str(random_seed) + '\n')
file_handle.write("Descriptor " + str(descriptor_choice) + '\n')
if key_control == 1:
    for n_pyd in range(len(pyd_phi)):
        file_handle.write('pyd_U:' + '\n')
        file_handle.write((str(pyd_U[n_pyd]) + " "))
    file_handle.write('\n')
    for n_pyd in range(len(pyd_phi)):
        file_handle.write('pyd_V:' + '\n')
        file_handle.write((str(pyd_V[n_pyd]) + " "))
    file_handle.write('\n')
file_handle.write('Iteration:' + '\n')
for n_iter in range(iterations):
    file_handle.write((str(n_iter) + " "))
    for idx in range(len(pyd)):
        file_handle.write(str(loss_history[n_iter, idx]) + " ")
    file_handle.write('\n')

torch.save(lmcn, debug_dir + '/optimization/opt_model.py')
torch.save(lmcn.state_dict(), debug_dir + '/optimization/params.pytorch')
