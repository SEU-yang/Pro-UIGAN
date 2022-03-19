# coding=gbk

import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log10
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import Image
#from tensorboardX import SummaryWriter
from models.network import *
from models.function import *
from data.datasets import *
import imgaug as ia
import imgaug.augmenters as iaa


os.makedirs("images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)
os.makedirs("loss_recoder", exist_ok=True)
os.makedirs("new_test_results/all", exist_ok=True)
os.makedirs("new_test_results/mask-128", exist_ok=True)
os.makedirs("new_test_results/oc-lr-16", exist_ok=True)
os.makedirs("new_test_results/sr-32", exist_ok=True)
os.makedirs("new_test_results/sr-64", exist_ok=True)
os.makedirs("new_test_results/sr-128", exist_ok=True)
os.makedirs("new_test_results/oc-lr-128", exist_ok=True)
os.makedirs("new_test_results/GT", exist_ok=True)
os.makedirs("new_test_results/oc-hr-128", exist_ok=True)
os.makedirs("new_test_results/input_landmarks", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--image_dir", type=str, default='training_list.txt', help="The path of the training dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--test_image_dir", type=str, default='testing_list.txt', help="The path of the training dataset")
parser.add_argument("--test_batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--imsize", type=int, default=128, help="image size")
parser.add_argument("--hr_height", type=int, default=128, help="hr_height")
parser.add_argument("--hr_width", type=int, default=128, help="hr_width")
parser.add_argument("--channel", type=int, default=3, help="channel")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=1, help="interval between model checkpoints")
parser.add_argument("--save_path", type=str, default="loss_recoder")
parser.add_argument('--image_size', type=int, default=128)
parser.add_argument('--N', type=int, default=10000)
parser.add_argument('--save_dir', type=str, default='masks_two')
#parser.add_argument('--vgg', type=str, default='experiments/vgg_normalised.pth')
parser.add_argument('--UInet1', type=str, default='saved_models/Pro_UInet.UInet160.pth')
parser.add_argument('--UInet2', type=str, default='saved_models/Pro_UInet.UInet260.pth')
parser.add_argument('--UInet3', type=str, default='saved_models/Pro_UInet.UInet360.pth')

    
opt = parser.parse_args()

global_shape = (128, 128)
local_shape = (64, 64)
imsize = opt.imsize


class ImageDataset(Dataset):
    def __init__(self, txt_path, lr_transforms_16, lr_transforms_32, lr_transforms_64, hr_transforms_128, hr_transforms_256):
        self.lr_transform_16 = transforms.Compose(lr_transforms_16)
        self.lr_transform_32 = transforms.Compose(lr_transforms_32)
        self.lr_transform_64 = transforms.Compose(lr_transforms_64)
        self.hr_transform_128 = transforms.Compose(hr_transforms_128)
        self.hr_transform_256 = transforms.Compose(hr_transforms_256)
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        self.img1_list = [i.split()[0] for i in lines]  # 得到所有的serve-ill image name
        self.img2_list = [i.split()[1] for i in lines]

    def __getitem__(self, idx):  # 根据 idx 取出其中一个
        img = Image.open(self.img1_list[idx % len(self.img1_list)])
        GT = Image.open(self.img2_list[idx % len(self.img2_list)])
        img_lr16 = self.lr_transform_16(GT)
        img_lr32 = self.lr_transform_32(GT)
        img_lr64 = self.lr_transform_64(GT)
        img_hr128 = self.hr_transform_128(GT)
        oc_lr = self.lr_transform_16(img)
        img_hr256 = self.hr_transform_256(GT)

        return {'lr16': img_lr16, 'lr32': img_lr32, 'lr64': img_lr64, 'hr128': img_hr128, 'hr256': img_hr256, 'input_lr': oc_lr}

    def __len__(self):
        return len(self.img1_list)


lr_transforms_16 = [transforms.Resize((imsize // 8, imsize // 8)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]

lr_transforms_32 = [transforms.Resize((imsize // 4, imsize // 4)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]

lr_transforms_64 = [transforms.Resize((imsize // 2, imsize // 2)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                    ]

hr_transforms_128 = [transforms.Resize((imsize, imsize)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                     ]

hr_transforms_256 = [transforms.Resize((imsize * 2, imsize * 2)),
                     transforms.ToTensor(),
                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                     ]


train_loader = Data.DataLoader(ImageDataset(opt.image_dir,
                                            lr_transforms_16=lr_transforms_16,
                                            lr_transforms_32=lr_transforms_32,
                                            lr_transforms_64=lr_transforms_64,
                                            hr_transforms_128=hr_transforms_128,
                                            hr_transforms_256=hr_transforms_256),                                            
                               batch_size=opt.batch_size, shuffle=True, num_workers=1)


test_loader = Data.DataLoader(ImageDataset(opt.test_image_dir,
                                           lr_transforms_16=lr_transforms_16,
                                           lr_transforms_32=lr_transforms_32,
                                           lr_transforms_64=lr_transforms_64,
                                           hr_transforms_128=hr_transforms_128,
                                           hr_transforms_256=hr_transforms_256),                                          
                              batch_size=opt.test_batch_size, shuffle=False, num_workers=1)


#### Define Network
cuda = True if torch.cuda.is_available() else False
uplr_2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
uplr_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
uplr_8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
uplr_16 = nn.Upsample(scale_factor=16, mode='bilinear', align_corners=True)
feature_extractor = FeatureExtractor()
Style_loss = InpaintingLoss(VGG16FeatureExtractor())
Global_D = Discriminator_global(input_shape=(opt.channel, opt.imsize, opt.imsize))
#Global_D = Discriminator_global(input_shape=(opt.channel, global_shape))
#Local_D = Discriminator_local(input_shape=(opt.channel, local_shape))
FAN_heatmap = FAN_heatmap()
FAN_loss = FAN_loss()
#### Initialize Network
Global_D._initialize_weights()


#### Define Generator
UInet1 = UInet11()
UInet2 = UInet22()
UInet3 = UInet33()


#### Initialize Network
#UInet1._initialize_weights()
UInet1.load_state_dict(torch.load(opt.UInet1))
#UInet1 = nn.Sequential(*list(UInet1.children())[:31])
UInet2.load_state_dict(torch.load(opt.UInet2))
UInet3.load_state_dict(torch.load(opt.UInet3))
#UInet2._initialize_weights()
#UInet3._initialize_weights()


#### Define Generator
Pro_UInet = Generator(UInet1, UInet2, UInet3)


#### Define Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()


if cuda:
    UInet1 = UInet1.cuda()
    UInet2 = UInet2.cuda()
    UInet3 = UInet3.cuda()
    Pro_UInet = Pro_UInet.cuda()
    Global_D = Global_D.cuda()
    #Local_D = Local_D.cuda()
    FAN_heatmap = FAN_heatmap.cuda()
    FAN_loss = FAN_loss.cuda()
    Style_loss = Style_loss.cuda()
    feature_extractor = feature_extractor.cuda()
    criterion_GAN = criterion_GAN.cuda()
    criterion_content = criterion_content.cuda()
    uplr_2 = uplr_2.cuda()
    uplr_4 = uplr_4.cuda()
    uplr_8 = uplr_8.cuda()
    uplr_16 = uplr_16.cuda()
    
    
avg_psnr = 0

# --------------test-------------
for i, imgs in enumerate(test_loader):
    print("Current batch : {}".format(i))

    # 'lr32': img_lr32, 'lr64': img_lr64, 'hr128': img_hr128, 'hr256': img_hr256, 'input_lr': input_oc_lr
    ## Data preprocess
    imgs_lr16 = imgs['lr16'].cuda()
    imgs_lr32 = imgs['lr32'].cuda()
    imgs_lr64 = imgs['lr64'].cuda()
    imgs_hr128 = imgs['hr128'].cuda()
    imgs_GT = imgs['hr128'].cuda()
    imgs_256 = imgs['hr256'].cuda()
    input_lrface = imgs['input_lr'].cuda()
    

    #######  INPUT Occluded-LR face        
    input_lrup = uplr_8(input_lrface)


    #######  INPUT LR Heatmaps
    input_lrup16 = uplr_16(input_lrface)
    heatmaps_lr = FAN_heatmap(input_lrup16)
    heatmaps_16_lr = F.interpolate(heatmaps_lr, size=16, mode='bilinear', align_corners=True)

    left_eye1_lr = heatmaps_16_lr[:, 36:37, :, :]
    left_eye2_lr = heatmaps_16_lr[:, 39:40, :, :]
    right_eye1_lr = heatmaps_16_lr[:, 42:43, :, :]
    right_eye2_lr = heatmaps_16_lr[:, 45:46, :, :]
    nose_lr =heatmaps_16_lr[:, 30:31, :, :]
    mouth1_lr = heatmaps_16_lr[:, 48:49, :, :]
    mouth2_lr = heatmaps_16_lr[:, 54:55, :, :]

    # INPUT LR 7 key heatmaps
    input_heatmaps_16_7 = torch.cat([left_eye1_lr, left_eye2_lr, right_eye1_lr, right_eye2_lr, nose_lr, mouth1_lr, mouth2_lr], 1)
    
    input_heatmaps_16_7_all = left_eye1_lr + left_eye2_lr + right_eye1_lr + right_eye2_lr + nose_lr + mouth1_lr + mouth2_lr
    
    input_heatmaps_16_7_all_rgb = torch.cat([input_heatmaps_16_7_all, input_heatmaps_16_7_all, input_heatmaps_16_7_all], 1)
    
    input_heatmaps_16_7_all_rgb = F.interpolate(input_heatmaps_16_7_all_rgb, size=128, mode='bilinear', align_corners=True)
       
    ###### Generator
    lr_face_out_16, landmarks_en_out_16, lr_face_out_32, landmarks_en_out_32, lr_face_out_64, landmarks_en_out_64, sr_face_32, sr_face_64, sr_face_128 = Pro_UInet(input_lrface, input_heatmaps_16_7)

 
    ### Output landmarks
    landmarks_en_out_16_all = landmarks_en_out_16[:, 0:1, :, :] + landmarks_en_out_16[:, 1:2, :, :] + landmarks_en_out_16[:, 2:3, :, :] + landmarks_en_out_16[:, 3:4, :, :] + landmarks_en_out_16[:, 4:5, :, :] + landmarks_en_out_16[:, 5:6, :, :] + landmarks_en_out_16[:, 6:7, :, :]
    
    landmarks_en_out_16_all_rgb = torch.cat([landmarks_en_out_16_all, landmarks_en_out_16_all, landmarks_en_out_16_all], 1)
    
    landmarks_en_out_16_all_rgb = F.interpolate(landmarks_en_out_16_all_rgb, size=128, mode='bilinear', align_corners=True)


    mse = criterion_GAN(sr_face_128.data, imgs_GT.data)
    #print(mse)
    psnr = 10 * log10(1 / mse)
    #print(psnr)
    avg_psnr = avg_psnr + psnr

    batches_done = i

    # Save image sample
    save_image(torch.cat((input_lrface.data, lr_face_out_16.data, imgs_lr16.data), -1),
               'new_test_results/all/a%d.png' % batches_done, normalize=True)
    save_image(torch.cat((sr_face_32.data, lr_face_out_32.data, imgs_lr32.data), -1),
               'new_test_results/all/b%d.png' % batches_done, normalize=True)
    save_image(torch.cat((sr_face_64.data, lr_face_out_64.data, imgs_lr64.data), -1),
               'new_test_results/all/c%d.png' % batches_done, normalize=True)
    save_image(torch.cat((input_lrup.data, sr_face_128.data, imgs_GT.data), -1),
               'new_test_results/all/d%d.png' % batches_done, normalize=True)
     
     
    save_image(input_heatmaps_16_7_all_rgb.data, 'new_test_results/input_landmarks/in%d.png' % batches_done,
               normalize=True)
    save_image(landmarks_en_out_16_all_rgb.data, 'new_test_results/input_landmarks/out%d.png' % batches_done,
               normalize=True)
                                                   
               
    save_image(input_lrup.data, 'new_test_results/oc-lr-128/%d.png' % batches_done,
               normalize=True)
    save_image(sr_face_32.data, 'new_test_results/sr-32/%d.png' % batches_done,
               normalize=True)
    save_image(sr_face_64.data, 'new_test_results/sr-64/%d.png' % batches_done,
               normalize=True)
    save_image(sr_face_128.data, 'new_test_results/sr-128/%03d.png' % batches_done,
               normalize=True)
    save_image(imgs_GT.data, 'new_test_results/GT/%d.png' % batches_done,
               normalize=True)
    save_image(input_lrface.data, 'new_test_results/oc-lr-16/%d.png' % batches_done,
               normalize=True)
    #save_image(input_hrface.data, 'new_test_results/oc-hr-128/%d.png' % batches_done,
               #normalize=True)
               

avg = avg_psnr / len(test_loader)
avg_all = "[avg_psnr: {:f}]".format(avg)

with open("Test_PSNR.txt", 'a') as f:
    f.write(avg_all + "\n")

