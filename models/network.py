import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import *
from models.function import *


class SANet(nn.Module):
    def __init__(self, input, in_planes):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(input, in_planes, (1, 1))
        self.g = nn.Conv2d(input, in_planes, (1, 1))
        self.h = nn.Conv2d(input, in_planes, (1, 1))
        self.k = nn.Conv2d(input, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.out_face = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.out_land = nn.Conv2d(in_planes, in_planes, (1, 1))

    def forward(self, content, style):
        F0 = self.f(content)  ## face image: 3*32*32  --  32*32*32
        G0 = self.g(style)    ## 68*32*32  --  32*32*32
        b, c, h, w = F0.size()    ## 32*32*32
        F = F0.view(b, -1, w * h).permute(0, 2, 1)  ## 1024*32
        G = G0.view(b, -1, w * h)   ## 32*1024
        S = torch.bmm(F, G)     ## 1024*1024
        S = self.sm(S)       ## 1024*1024

        Face_value = self.h(content).view(b, -1, w * h)  # 32*1024
        Face = torch.bmm(Face_value, S)  ## 32*1024
        #Face = torch.bmm(Face_value, S.permute(0, 2, 1))  ## 32*1024
        Face_en = Face.view(b, c, h, w)   ## 32*32*32
        Face_en += G0  ## face attention features + landmark features 32*32*32
        enhanced_landmarks = Face_en

        Landmark_value = self.k(style).view(b, -1, w * h)  # B X C X N
        Landmark = torch.bmm(Landmark_value, S.permute(0, 2, 1))   ## 32*1024
        #Landmark = torch.bmm(Landmark_value, S)  ## 32*1024
        Landmark_en = Landmark.view(b, c, h, w)    ## 32*32*32
        Landmark_en += F0     ## landmark attention features + face features 32*32*32
        enhanced_facefeatures = Landmark_en
        return enhanced_facefeatures, enhanced_landmarks


### define Generator and Discriminator
class CrosslevelNet1(nn.Module):
    def __init__(self):
        super(CrosslevelNet1, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())

        self.conv1_2 = nn.Sequential(nn.Conv2d(7, 128, 3, 1, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())

        self.cross_level_1 = SANet(128, 32)

        self.conv_out11 = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1),
                                     nn.Tanh())
        self.conv_out12 = nn.Sequential(nn.Conv2d(32, 7, 3, 1, 1),
                                       nn.Sigmoid())

    def forward(self, lr_face, landmarks):
            lr_face_in = self.conv1_1(lr_face)
            landmarks_in = self.conv1_2(landmarks)
            lr_face_en, landmarks_en = self.cross_level_1(lr_face_in, landmarks_in)
            lr_face_out_1 = self.conv_out11(lr_face_en)
            landmarks_en_out_1 = self.conv_out12(landmarks_en)
            return lr_face_out_1, landmarks_en_out_1   ## B*7*16*16


class CrosslevelNet2(nn.Module):
    def __init__(self):
        super(CrosslevelNet2, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU())

        self.conv1_2 = nn.Sequential(nn.Conv2d(42, 128, 3, 1, 1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU())

        self.cross_level_2 = SANet(128, 32)

        self.conv_out11 = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1),
                                     nn.Tanh())
        self.conv_out12 = nn.Sequential(nn.Conv2d(32, 42, 3, 1, 1),
                                       nn.Sigmoid())

    def forward(self, lr_face, landmarks):
        lr_face_in = self.conv1_1(lr_face)
        landmarks_in = self.conv1_2(landmarks)
        lr_face_en, landmarks_en = self.cross_level_2(lr_face_in, landmarks_in)
        lr_face_out_1 = self.conv_out11(lr_face_en)
        landmarks_en_out_1 = self.conv_out12(landmarks_en)
        return lr_face_out_1, landmarks_en_out_1     ## B*42*32*32


class CrosslevelNet3(nn.Module):
    def __init__(self):
        super(CrosslevelNet3, self).__init__()
        self.conv1_1 = nn.Sequential(nn.Conv2d(3, 128, 3, 1, 1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU())

        self.conv1_2 = nn.Sequential(nn.Conv2d(68, 128, 3, 1, 1),
                                     nn.BatchNorm2d(128),
                                     nn.ReLU())

        self.cross_level_3 = SANet(128, 32)

        self.conv_out11 = nn.Sequential(nn.Conv2d(32, 3, 3, 1, 1),
                                     nn.Tanh())
        self.conv_out12 = nn.Sequential(nn.Conv2d(32, 68, 3, 1, 1),
                                       nn.Sigmoid())

    def forward(self, lr_face, landmarks):
        lr_face_in = self.conv1_1(lr_face)
        landmarks_in = self.conv1_2(landmarks)
        lr_face_en, landmarks_en = self.cross_level_3(lr_face_in, landmarks_in)
        lr_face_out_1 = self.conv_out11(lr_face_en)
        landmarks_en_out_1 = self.conv_out12(landmarks_en)
        return lr_face_out_1, landmarks_en_out_1    ## B*68*64


class SRnet1(nn.Module):
    def __init__(self):
        super(SRnet1, self).__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(3 + 7, 128, 3, 1, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())

        self.conv1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())

        # Upsampling layers 32*32 -- 64*64
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        res_blocks = []
        for _ in range(2):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)


        self.conv3 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU()
                                   )

        self.conv4 = nn.Sequential(nn.Conv2d(32, 12, 3, 1, 1),
                                   nn.BatchNorm2d(12),
                                   nn.ReLU()
                                   )

        self.conv5 = nn.Sequential(nn.Conv2d(12, 3, 3, 1, 1),
                                   nn.Tanh())

    def forward(self, lr_face, landmarks):
        input = torch.cat((lr_face, landmarks), dim=1)  # (3+7)*32*32
        lr_face_out128 = self.conv0(input)  # 128*32*32
        lr_face_out2 = self.conv1(lr_face_out128)  # 64*32*32
        x64 = self.up1(lr_face_out2)  # 64*64*64
        x64 = self.res_blocks(x64)  # 64*128*128
        #x128 = self.up2(x64)  # 64*128*128
        x3 = self.conv3(x64)  # 32*128*128
        x4 = self.conv4(x3)  # 12*128*128
        out = self.conv5(x4)  # 3*128*128
        return out  # 3*128*128


class SRnet2(nn.Module):
     def __init__(self):
        super(SRnet2, self).__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(3 + 42, 128, 3, 1, 1),
                                         nn.BatchNorm2d(128),
                                         nn.ReLU())

        self.conv1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())

        # Upsampling layers 32*32 -- 64*64
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        res_blocks = []
        for _ in range(2):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)


        self.conv3 = nn.Sequential(nn.Conv2d(64, 32, 5, 1, 2),
                        nn.BatchNorm2d(32),
                        nn.ReLU()
                       )
                       
        self.conv4 = nn.Sequential(nn.Conv2d(32, 12, 5, 1, 2),
                        nn.BatchNorm2d(12),
                        nn.ReLU()
                       )
        
        self.conv5 = nn.Sequential(nn.Conv2d(12, 3, 3, 1, 1),
                                     nn.Tanh())

        
     def forward(self, lr_face, landmarks):
            input = torch.cat((lr_face, landmarks), dim=1)   # (3+42)*32*32
            lr_face_out128 = self.conv0(input)    # 128*32*32
            lr_face_out2 = self.conv1(lr_face_out128)   # 64*32*32
            x64 = self.up1(lr_face_out2)  # 64*64*64
            x64 = self.res_blocks(x64)  # 64*128*128
            x3=self.conv3(x64)  # 32*128*128
            x4=self.conv4(x3)   # 12*128*128
            out=self.conv5(x4)   # 3*128*128
            return out   #3*128*128


class SRnet3(nn.Module):
    def __init__(self):
        super(SRnet3, self).__init__()

        self.conv0 = nn.Sequential(nn.Conv2d(3 + 68, 128, 3, 1, 1),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU())

        self.conv1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU())

        # Upsampling layers 32*32 -- 64*64
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        res_blocks = []
        for _ in range(2):
            res_blocks.append(ResidualBlock(64))
        self.res_blocks = nn.Sequential(*res_blocks)


        self.conv3 = nn.Sequential(nn.Conv2d(64, 32, 5, 1, 2),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU()
                                   )

        self.conv4 = nn.Sequential(nn.Conv2d(32, 12, 5, 1, 2),
                                   nn.BatchNorm2d(12),
                                   nn.ReLU()
                                   )

        self.conv5 = nn.Sequential(nn.Conv2d(12, 3, 3, 1, 1),
                                   nn.Tanh())

    def forward(self, lr_face, landmarks):
        input = torch.cat((lr_face, landmarks), dim=1)  # (3+68)*32*32
        lr_face_out128 = self.conv0(input)  # 128*32*32
        lr_face_out2 = self.conv1(lr_face_out128)  # 64*32*32
        x64 = self.up1(lr_face_out2)  # 64*64*64
        x64 = self.res_blocks(x64)  # 64*128*128
        #x128 = self.up2(x64)  # 64*128*128
        x3 = self.conv3(x64)  # 32*128*128
        x4 = self.conv4(x3)  # 12*128*128
        out = self.conv5(x4)  # 3*128*128
        return out  # 3*128*128


### define Generator and Discriminator
class UInet1(nn.Module):
    def __init__(self):
        super(UInet1, self).__init__()
        self.CLnet_32 = CrosslevelNet1()
        self.SRnet_32 = SRnet1()

    def forward(self, lr_face, landmarks_16):
            lr_face_out_1, landmarks_en_out_1 = self.CLnet_32(lr_face, landmarks_16)
            SR_face_32 = self.SRnet_32(lr_face_out_1, landmarks_en_out_1)
            return lr_face_out_1, landmarks_en_out_1, SR_face_32


### define Generator and Discriminator
class UInet2(nn.Module):
    def __init__(self):
        super(UInet2, self).__init__()
        self.CLnet_64 = CrosslevelNet2()
        self.SRnet_64 = SRnet2()


    def forward(self, lr_face, landmarks_32):
            lr_face_out_2, landmarks_en_out_2 = self.CLnet_64(lr_face, landmarks_32)
            SR_face_64 = self.SRnet_64(lr_face_out_2, landmarks_en_out_2)
            return lr_face_out_2, landmarks_en_out_2, SR_face_64


### define Generator and Discriminator
class UInet3(nn.Module):
    def __init__(self):
        super(UInet3, self).__init__()
        self.CLnet_128 = CrosslevelNet3()
        self.SRnet_128 = SRnet3()

    def forward(self, lr_face,  landmarks_64):
            lr_face_out_3, landmarks_en_out_3 = self.CLnet_128(lr_face, landmarks_64)
            SR_face_128 = self.SRnet_128(lr_face_out_3, landmarks_en_out_3)
            return lr_face_out_3, landmarks_en_out_3, SR_face_128


cuda = True if torch.cuda.is_available() else False
FAN_heatmap = FAN_heatmap()

if cuda:
    FAN_heatmap = FAN_heatmap.cuda()


### define Generator and Discriminator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net1 = UInet1()
        self.net2 = UInet2()
        self.net3 = UInet3()

    def forward(self, lr_face, landmarks_16):
            ####  UI-block1
            lr_face_out_1, landmarks_en_out_1, SR_face_32 = self.net1(lr_face, landmarks_16)

            ####  Calculate 42 input landmark heatmaps
            SR_face_32_x8 = F.interpolate(SR_face_32, size=256, mode='bilinear', align_corners=True)
            heatmaps_64_SR_1 = FAN_heatmap(SR_face_32_x8)
            heatmaps_32_SR_1 = F.interpolate(heatmaps_64_SR_1, size=32, mode='bilinear', align_corners=True)

            # 42 key heatmaps
            gl_32 = heatmaps_32_SR_1[:, 0:17, :, :]
            mei_32_1 = heatmaps_32_SR_1[:, 17:22, :, :]
            mei_32_2 = heatmaps_32_SR_1[:, 22:27, :, :]
            left_eye_32_1 = heatmaps_32_SR_1[:, 36:37, :, :]
            left_eye_32_2 = heatmaps_32_SR_1[:, 39:40, :, :]
            right_eye_32_1 = heatmaps_32_SR_1[:, 42:43, :, :]
            right_eye_32_2 = heatmaps_32_SR_1[:, 45:46, :, :]
            nose_32 = heatmaps_32_SR_1[:, 27:31, :, :]
            mouth_32_1 = heatmaps_32_SR_1[:, 48:49, :, :]
            mouth_32_2 = heatmaps_32_SR_1[:, 54:55, :, :]
            mouth_32 = heatmaps_32_SR_1[:, 60:65, :, :]

            # 42 key heatmaps
            heatmaps_32_SR_42 = torch.cat(
                [left_eye_32_1, left_eye_32_2, right_eye_32_1, right_eye_32_2, nose_32, mouth_32_1, mouth_32_2, gl_32,
                 mei_32_1, mei_32_2, mouth_32], 1)

            ####  UI-block2
            lr_face_out_2, landmarks_en_out_2, SR_face_64 = self.net2(SR_face_32, heatmaps_32_SR_42)

            SR_face_64_x4 = F.interpolate(SR_face_64, size=256, mode='bilinear', align_corners=True)
            heatmaps_64_SR_2 = FAN_heatmap(SR_face_64_x4)

            ####  UI-block3c
            lr_face_out_3, landmarks_en_out_3, SR_face_128 = self.net3(SR_face_64, heatmaps_64_SR_2)
            
            return lr_face_out_1, landmarks_en_out_1, lr_face_out_2, landmarks_en_out_2, lr_face_out_3, landmarks_en_out_3, SR_face_32, SR_face_64, SR_face_128

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.normal_(0.0, 0.02)
                if isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0.0, 0.02)
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.data.normal_(1.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.normal_(0.0, 0.02)


class Discriminator_global(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator_global, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 4), int(in_width / 2 ** 4)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        out_2 = self.model(img)
        out = self.sigmoid(out_2)
        return out

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.normal_(0.0, 0.02)
                if isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0.0, 0.02)
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.data.normal_(1.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.normal_(0.0, 0.02)


class Discriminator_local(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator_local, self).__init__()

        self.input_shape = input_shape
        in_channels, in_height, in_width = self.input_shape
        patch_h, patch_w = int(in_height / 2 ** 3), int(in_width / 2 ** 3)
        self.output_shape = (1, patch_h, patch_w)

        def discriminator_block(in_filters, out_filters, first_block=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers = []
        in_filters = in_channels
        for i, out_filters in enumerate([64, 128, 256]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0)))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, img):
        return self.model(img)

    def _initialize_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.normal_(0.0, 0.02)
                if isinstance(m, nn.ConvTranspose2d):
                    m.weight.data.normal_(0.0, 0.02)
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.data.normal_(1.0, 0.02)
                    if m.bias is not None:
                        m.bias.data.normal_(0.0, 0.02)
