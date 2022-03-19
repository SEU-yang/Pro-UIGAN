import torch
import torch.utils.data as Data
import glob
import random
import os
import numpy as np
import time

import torchvision.transforms as transforms
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.models import vgg19
from torchvision.models import vgg16
import math
import sys
import cv2
from math import floor
# import dlib
import torchvision
from models.FAN import FAN
from torch.utils.model_zoo import load_url


def adjust_learning_rate(optimizer, epoch, lrr):
    lr = lrr * (0.99 ** (epoch // 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16_model = vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16_model.features[:5])
        self.enc_2 = nn.Sequential(*vgg16_model.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16_model.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        # print(results.shape)
        out = results[1:]
        return out


def gram_matrix(feat):
    # https://github.com/pytorch/examples/blob/master/fast_neural_style/neural_style/utils.py
    (b, ch, h, w) = feat.size()
    feat = feat.view(b, ch, h * w)
    feat_t = feat.transpose(1, 2)
    gram = torch.bmm(feat, feat_t) / (ch * h * w)
    return gram


def total_variation_loss(image):
    # shift one pixel and get difference (for both x and y direction)
    loss = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) + \
           torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))
    return loss


class InpaintingLoss(nn.Module):
    def __init__(self, extractor):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.extractor = extractor

    def forward(self, output, gt):
        loss_dict = {}
        # loss_dict['hole'] = self.l1((1 - mask) * output, (1 - mask) * gt)
        # loss_dict['valid'] = self.l1(mask * output, mask * gt)

        if output.shape[1] == 3:
            feat_output = self.extractor(output)
            feat_gt = self.extractor(gt)
        elif output.shape[1] == 1:
            feat_output = self.extractor(torch.cat([output] * 3, 1))
            feat_gt = self.extractor(torch.cat([gt] * 3, 1))
        else:
            raise ValueError('only gray an')

        loss_dict['style'] = 0.0
        for i in range(3):
            loss_dict['style'] += self.l1(gram_matrix(feat_output[i]),
                                          gram_matrix(feat_gt[i]))

        #loss_dict['style'] = self.l1(gram_matrix(feat_output[0]), gram_matrix(feat_gt[0])) + self.l1(gram_matrix(feat_output[1]), gram_matrix(feat_gt[1])) + self.l1(gram_matrix(feat_output[2]), gram_matrix(feat_gt[2]))

        loss_dict['tv'] = total_variation_loss(output)
        l_style = loss_dict['style']
        l_tv = loss_dict['tv']
        return l_style, l_tv


# dlib 68 landmark predict
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor('pretrained_models/shape_predictor_68_face_landmarks.dat')


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


def normal(feat, eps=1e-5):
    feat_mean, feat_std = calc_mean_std(feat, eps)
    normalized = (feat - feat_mean) / feat_std
    return normalized


class FAN_loss(nn.Module):
    def __init__(self):
        super(FAN_loss, self).__init__()
        FAN_net = FAN(4)
        FAN_model_url = 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar'
        fan_weights = load_url(FAN_model_url, map_location=lambda storage, loc: storage)
        FAN_net.load_state_dict(fan_weights)
        for p in FAN_net.parameters():
            p.requires_grad = False
        self.FAN_net = FAN_net
        self.criterion = nn.MSELoss()

    def forward(self, data, target):
        heat_predict = self.FAN_net(data)
        heat_gt = self.FAN_net(target)
        loss = self.criterion(self.FAN_net(data)[0], self.FAN_net(target)[0])
        # print(data[0].size())
        # print(target[0].size())
        # exit()
        return loss


class FAN_heatmap(nn.Module):
    def __init__(self):
        super(FAN_heatmap, self).__init__()
        FAN_net = FAN(4)
        FAN_model_url = 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar'
        fan_weights = load_url(FAN_model_url, map_location=lambda storage, loc: storage)
        FAN_net.load_state_dict(fan_weights)
        for p in FAN_net.parameters():
            p.requires_grad = False
        self.FAN_net = FAN_net

    def forward(self, data):
        heat_gt = self.FAN_net(data)
        return heat_gt[0]


def gen_input_mask(
        shape, shapeRGB, hole_size,
        hole_area=None, max_holes=1):
    """
    * inputs:
        - shape (sequence, required):
                Shape of a mask tensor to be generated.
                A sequence of length 4 (N, C, H, W) is assumed.
        - hole_size (sequence or int, required):
                Size of holes created in a mask.
                If a sequence of length 4 is provided,
                holes of size (W, H) = (
                    hole_size[0][0] <= hole_size[0][1],
                    hole_size[1][0] <= hole_size[1][1],
                ) are generated.
                All the pixel values within holes are filled with 1.0.
        - hole_area (sequence, optional):
                This argument constraints the area where holes are generated.
                hole_area[0] is the left corner (X, Y) of the area,
                while hole_area[1] is its width and height (W, H).
                This area is used as the input region of Local discriminator.
                The default value is None.
        - max_holes (int, optional):
                This argument specifies how many holes are generated.
                The number of holes is randomly chosen from [1, max_holes].
                The default value is 1.
    * returns:
            A mask tensor of shape [N, C, H, W] with holes.
            All the pixel values within holes are filled with 1.0,
            while the other pixel values are zeros.
    """
    mask = torch.zeros(shape)
    mask_random = torch.zeros(shapeRGB)
    bsize, _, mask_h, mask_w = mask.shape
    for i in range(bsize):
        n_holes = random.choice(list(range(1, max_holes + 1)))
        for _ in range(n_holes):
            # choose patch width
            if isinstance(hole_size[0], tuple) and len(hole_size[0]) == 2:
                hole_w = random.randint(hole_size[0][0], hole_size[0][1])
            else:
                hole_w = hole_size[0]

            # choose patch height
            if isinstance(hole_size[1], tuple) and len(hole_size[1]) == 2:
                hole_h = random.randint(hole_size[1][0], hole_size[1][1])
            else:
                hole_h = hole_size[1]

            # choose offset upper-left coordinate
            if hole_area:
                harea_xmin, harea_ymin = hole_area[0]
                harea_w, harea_h = hole_area[1]
                offset_x = random.randint(harea_xmin, harea_xmin + harea_w - hole_w)
                offset_y = random.randint(harea_ymin, harea_ymin + harea_h - hole_h)
            else:
                offset_x = random.randint(0, mask_w - hole_w)
                offset_y = random.randint(0, mask_h - hole_h)
            mask[i, :, offset_y: offset_y + hole_h, offset_x: offset_x + hole_w] = 1.0
            mask_random[i, :, offset_y: offset_y + hole_h, offset_x: offset_x + hole_w] = torch.from_numpy((np.random.rand(3, hole_h, hole_w)-0.5)/0.5)
    return mask, mask_random


def gen_hole_area(size, mask_size):
    """
    * inputs:
        - size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of hole area.
        - mask_size (sequence, required)
                A sequence of length 2 (W, H) is assumed.
                (W, H) is the size of input mask.
    * returns:
            A sequence used for the input argument 'hole_area' for function 'gen_input_mask'.
    """
    mask_w, mask_h = mask_size
    harea_w, harea_h = size
    offset_x = random.randint(0, mask_w - harea_w)
    offset_y = random.randint(0, mask_h - harea_h)
    return ((offset_x, offset_y), (harea_w, harea_h))


def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A torch tensor of shape (N, C, H, W) is assumed.
        - area (sequence, required)
                A sequence of length 2 ((X, Y), (W, H)) is assumed.
                sequence[0] (X, Y) is the left corner of an area to be cropped.
                sequence[1] (W, H) is its width and height.
    * returns:
            A torch tensor of shape (N, C, H, W) cropped in the specified area.
    """
    xmin, ymin = area[0]
    w, h = area[1]
    return x[:, :, ymin: ymin + h, xmin: xmin + w]


def sample_random_batch(dataset, batch_size=32):
    """
    * inputs:
        - dataset (torch.utils.data.Dataset, required)
                An instance of torch.utils.data.Dataset.
        - batch_size (int, optional)
                Batch size.
    * returns:
            A mini-batch randomly sampled from the input dataset.
    """
    num_samples = len(dataset)
    batch = []
    for _ in range(min(batch_size, num_samples)):
        index = random.choice(range(0, num_samples))
        x = torch.unsqueeze(dataset[index], dim=0)
        batch.append(x)
    return torch.cat(batch, dim=0)


def poisson_blend(x, output, mask):
    """
    * inputs:
        - x (torch.Tensor, required)
                Input image tensor of shape (N, 3, H, W).
        - output (torch.Tensor, required)
                Output tensor from Completion Network of shape (N, 3, H, W).
        - mask (torch.Tensor, required)
                Input mask tensor of shape (N, 1, H, W).
    * returns:
                An image tensor of shape (N, 3, H, W) inpainted
                using poisson image editing method.
    """
    x = x.clone().cpu()
    output = output.clone().cpu()
    mask = mask.clone().cpu()
    mask = torch.cat((mask, mask, mask), dim=1)  # convert to 3-channel format
    num_samples = x.shape[0]
    ret = []
    for i in range(num_samples):
        dstimg = transforms.functional.to_pil_image(x[i])
        dstimg = np.array(dstimg)[:, :, [2, 1, 0]]
        srcimg = transforms.functional.to_pil_image(output[i])
        srcimg = np.array(srcimg)[:, :, [2, 1, 0]]
        msk = transforms.functional.to_pil_image(mask[i])
        msk = np.array(msk)[:, :, [2, 1, 0]]
        # compute mask's center
        xs, ys = [], []
        for i in range(msk.shape[0]):
            for j in range(msk.shape[1]):
                if msk[i, j, 0] == 255:
                    ys.append(i)
                    xs.append(j)
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        center = ((xmax + xmin) // 2, (ymax + ymin) // 2)
        out = cv2.seamlessClone(srcimg, dstimg, msk, center, cv2.NORMAL_CLONE)
        out = out[:, :, [2, 1, 0]]
        out = transforms.functional.to_tensor(out)
        out = torch.unsqueeze(out, dim=0)
        ret.append(out)
    ret = torch.cat(ret, dim=0)
    return ret


def poisson_blend_old(input, output, mask):
    """
    * inputs:
        - input (torch.Tensor, required)
                Input tensor of Completion Network.
        - output (torch.Tensor, required)
                Output tensor of Completion Network.
        - mask (torch.Tensor, required)
                Input mask tensor of Completion Network.
    * returns:
                Image tensor inpainted using poisson image editing method.
    """
    num_samples = input.shape[0]
    ret = []

    # convert torch array to numpy array followed by
    # converting 'channel first' format to 'channel last' format.
    input_np = np.transpose(np.copy(input.cpu().numpy()), axes=(0, 2, 3, 1))
    output_np = np.transpose(np.copy(output.cpu().numpy()), axes=(0, 2, 3, 1))
    mask_np = np.transpose(np.copy(mask.cpu().numpy()), axes=(0, 2, 3, 1))

    # apply poisson image editing method for each input/output image and mask.
    for i in range(num_samples):
        inpainted_np = blend(input_np[i], output_np[i], mask_np[i])
        inpainted = torch.from_numpy(np.transpose(inpainted_np, axes=(2, 0, 1)))
        inpainted = torch.unsqueeze(inpainted, dim=0)
        ret.append(inpainted)
    ret = torch.cat(ret, dim=0)
    return ret


def crop(x, area):
    """
    * inputs:
        - x (torch.Tensor, required)
                A torch tensor of shape (N, C, H, W) is assumed.
        - area (sequence, required)
                A sequence of length 2 ((X, Y), (W, H)) is assumed.
                sequence[0] (X, Y) is the left corner of an area to be cropped.
                sequence[1] (W, H) is its width and height.
    * returns:
            A torch tensor of shape (N, C, H, W) cropped in the specified area.
    """
    xmin, ymin = area[0]
    w, h = area[1]
    return x[:, :, ymin: ymin + h, xmin: xmin + w]


def get_facial_landmarks(img):
    rects = detector(img, 0)
    # print(rects.shape)
    rect = rects[0]
    # ~ print(len(rects))
    # ~ print("22222")
    shape = predictor(img, rect)
    a = np.array([[pt.x, pt.y] for pt in shape.parts()])
    b = a.astype('float')
    return b


def _putGaussianMap(center, visible_flag, crop_size_y, crop_size_x, sigma):
    """
    根据一个中心点,生成一个heatmap
    :param center:
    :return:
    """
    grid_y = crop_size_y
    grid_x = crop_size_x
    if visible_flag == False:
        return np.zeros((grid_y, grid_x))
    # start = stride / 2.0 - 0.5
    y_range = [i for i in range(grid_y)]
    x_range = [i for i in range(grid_x)]
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx
    yy = yy
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    heatmap = np.exp(-exponent)
    return heatmap


def _putGaussianMaps(keypoints, crop_size_y, crop_size_x, sigma):
    """
    :param keypoints: (15,2)
    :param crop_size_y: int
    :param crop_size_x: int
    :param stride: int
    :param sigma: float
    :return:
    """
    all_keypoints = keypoints
    point_num = all_keypoints.shape[0]
    heatmaps_this_img = []
    for k in range(point_num):
        flag = ~np.isnan(all_keypoints[k, 0])
        heatmap = _putGaussianMap(all_keypoints[k], flag, crop_size_y, crop_size_x, sigma)
        heatmap = heatmap[np.newaxis, ...]
        heatmaps_this_img.append(heatmap)
    heatmaps_this_img = np.concatenate(heatmaps_this_img, axis=0)  # (num_joint,crop_size_y/stride,crop_size_x/stride)
    return heatmaps_this_img


def get_heatmaps(img):
    imgs_numpy = img.cpu().squeeze(0).permute(1, 2, 0)
    imgs_numpy = imgs_numpy.numpy()
    imgs_numpy = (imgs_numpy * 0.5 + 0.5) * 255
    imgs_numpy = imgs_numpy.astype(np.uint8)
    real_land = get_facial_landmarks(imgs_numpy)  # to tensor
    real_heatmaps = _putGaussianMaps(real_land, 128, 128, 5)
    real_heatmaps = torch.from_numpy(real_heatmaps)
    real_heatmaps = real_heatmaps.float()
    real_heatmaps = real_heatmaps.unsqueeze(0)
    real_heatmaps = real_heatmaps.cuda()
    return real_heatmaps


# def draw_features(width,height,x,savename):
#     tic=time.time()
#     fig = plt.figure(figsize=(16, 16))
#     fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=
# 0.05, hspace=0.05)
#     for i in range(width*height):
#         plt.subplot(height,width, i + 1)
#         plt.axis('off')
#         img = x[0, i, :, :]
#         #print(img.shape)
#         #pmin = np.min(img)
#         #pmax = np.max(img)
#         #img = ((img - pmin) / (pmax - pmin + 0.000001))*255  #float在[0，1] 之间，转换成0-255
#         #img=img.astype(np.uint8)  #转成unit8
#         #img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
#         #img = img[:, :, ::-1]#注意cv2（BGR）和matplotlib(RGB)通道是相反的
#         plt.imshow(img)
#         #print("{}/{}".format(i,width*height))
#     fig.savefig(savename, dpi=100)
#     fig.clf()


def draw_features(width, height, x, savename):
    tic = time.time()
    fig = plt.figure(figsize=(32, 32))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=
    0.05, hspace=0.05)
    for i in range(width * height):
        plt.subplot(height, width, i + 1)
        plt.axis('off')
        img = x[0, i, :, :]
        # print(img.shape)
        # pmin = np.min(img)
        # pmax = np.max(img)
        img = img * 255  # float在[0，1] 之间，转换成0-255
        img = img.astype(np.uint8)  # 转成unit8
        # img = cv2.applyColorMap(img, cv2.COLORMAP_JET)  # 生成heat map
        img = img[:, ::-1]  # 注意cv2（BGR）和matplotlib(RGB)通道是相反的
        plt.imshow(img)
        # print("{}/{}".format(i,width*height))
    fig.savefig(savename, dpi=80)
    fig.clf()


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img


def show_from_tensor(tensor, title=None):
    img = tensor.clone()
    img = tensor_to_np(img)
    plt.figure()
    plt.imshow(img)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)


unloader = transforms.ToPILImage()


def get_peak_points(heatmaps):
    """
    :param heatmaps: numpy array (N,15,96,96)
    :return:numpy array (N,15,2)
    """
    N,C,H,W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i,j] == heatmaps[i,j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x,y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points
    

five_pts_idx = [ [36,41] , [42,47] , [27,35] , [48,48] , [54,54] ]
def landmarks_68_to_5(x):
    y = []
    for j in range(5):
        y.append( np.mean( x[five_pts_idx[j][0]:five_pts_idx[j][1] + 1] , axis = 0  ) )
    return np.array( y , np.float32)  
    
     
def process(img, landmarks_5pts):
    batch = {}
    name = ['left_eye','right_eye','nose','mouth']
    patch_size = {
            'left_eye':(40,40),
            'right_eye':(40,40),
            'nose':(64,64),
            'mouth':(48,32),            
    }
    landmarks_5pts[3,0] =  (landmarks_5pts[3,0] + landmarks_5pts[4,0]) / 2.0
    landmarks_5pts[3,1] = (landmarks_5pts[3,1] + landmarks_5pts[4,1]) / 2.0

    # crops
    for i in range(4):
        print(i)
        x = floor(landmarks_5pts[i,0])
        y = floor(landmarks_5pts[i,1])
        batch[ name[i] ] = img.crop( (x - patch_size[ name[i] ][0]//2 + 1 , y - patch_size[ name[i] ][1]//2 + 1 , x + patch_size[ name[i] ][0]//2 + 1 , y + patch_size[ name[i] ][1]//2 + 1 ) )
    return batch
    
    