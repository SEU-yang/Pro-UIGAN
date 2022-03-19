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
os.makedirs("test_results/all", exist_ok=True)
os.makedirs("test_results/mask-128", exist_ok=True)
os.makedirs("test_results/lr-16", exist_ok=True)
os.makedirs("test_results/lr-32", exist_ok=True)
os.makedirs("test_results/lr-64", exist_ok=True)
os.makedirs("test_results/sr-128", exist_ok=True)
os.makedirs("test_results/oc-lr-16", exist_ok=True)
os.makedirs("test_results/oc-lr-128", exist_ok=True)
os.makedirs("test_results/GT", exist_ok=True)


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--image_dir", type=str, default="../Data/CelebaHQ", help="The path of the training dataset")
parser.add_argument("--batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--test_image_dir", type=str, default="../Data/celeba-test", help="The path of the training dataset")
parser.add_argument("--test_batch_size", type=int, default=4, help="size of the batches")
parser.add_argument("--lr_G", type=float, default=5e-4, help="adam: learning rate")
parser.add_argument("--lr_D", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
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

    
opt = parser.parse_args()

global_shape = (128, 128)
local_shape = (64, 64)
imsize = opt.imsize


class ImageDataset(Dataset):
    def __init__(self, root, lr_transforms_16, lr_transforms_32, lr_transforms_64, hr_transforms_128,
                 hr_transforms_256):
        self.hr_img_size = 128
        self.lr_transform_16 = transforms.Compose(lr_transforms_16)
        self.lr_transform_32 = transforms.Compose(lr_transforms_32)
        self.lr_transform_64 = transforms.Compose(lr_transforms_64)
        self.hr_transform_128 = transforms.Compose(hr_transforms_128)
        self.hr_transform_256 = transforms.Compose(hr_transforms_256)        
        self.files = sorted(glob.glob(root + '/*.*'))

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)])
        img_lr16 =  self.lr_transform_16(img)
        img_lr32 = self.lr_transform_32(img)
        img_lr64 = self.lr_transform_64(img)
        img_hr128 = self.hr_transform_128(img)
        img_hr256 = self.hr_transform_256(img)

        return {'lr16': img_lr16, 'lr32': img_lr32, 'lr64': img_lr64, 'hr128': img_hr128, 'hr256': img_hr256}

    def __len__(self):
        return len(self.files)


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
Pro_UInet = Generator()
feature_extractor = FeatureExtractor()
Style_loss = InpaintingLoss(VGG16FeatureExtractor())
Global_D = Discriminator_global(input_shape=(opt.channel, global_shape))
Local_D = Discriminator_local(input_shape=(opt.channel, local_shape))
FAN_heatmap = FAN_heatmap()
FAN_loss = FAN_loss()


Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
log_path = opt.save_path
#writer = SummaryWriter(log_dir=log_path)


#### Define Losses
criterion_GAN = torch.nn.MSELoss()
criterion_content = torch.nn.L1Loss()

if cuda:
    Pro_UInet = Pro_UInet.cuda()
    Global_D = Global_D.cuda()
    Local_D = Local_D.cuda()
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

# init_img = Tensor(1, 3, 16, 16)
# init_heatmap = Tensor(1, 68, 16, 16)
# #writer.add_graph(Generator_1, (init_img, init_heatmap))


#### Initialize Network
Pro_UInet._initialize_weights()
Global_D._initialize_weights()
Local_D._initialize_weights()

#### Define Optimizer
optimizer_G = torch.optim.Adam(Pro_UInet.parameters(), lr=opt.lr_G)
optimizer_Global_D = torch.optim.Adam(Global_D.parameters(), lr=opt.lr_D)
optimizer_Local_D = torch.optim.Adam(Local_D.parameters(), lr=opt.lr_D)


#### Start Training
for epoch in range(opt.epoch, opt.n_epochs):

    avg_psnr = 0
    print("Current epoch : {}".format(epoch))
    
    
    # --------------test-------------
    for i, imgs in enumerate(test_loader):
        print("Current batch : {}".format(i))

        # 'lr32': img_lr32, 'lr64': img_lr64, 'hr128': img_hr128, 'hr256': img_hr256
        ## Data preprocess
        imgs_lr16 = imgs['lr16'].cuda()
        imgs_lr32 = imgs['lr32'].cuda()
        imgs_lr64 = imgs['lr64'].cuda()
        imgs_hr128 = imgs['hr128'].cuda()
        imgs_GT = imgs['hr128'].cuda()
        imgs_256 = imgs['hr256'].cuda()
        

        mask_list = []
        mask_white_list = []
        for j in range(opt.batch_size):   
            canvas = np.zeros((opt.image_size, opt.image_size, 3))
            canvas_white = np.zeros((opt.image_size, opt.image_size))
            hole_size=((56, 64), (56, 64))
            action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            ini_x = random.randint(0, opt.image_size - 1)
            ini_y = random.randint(0, opt.image_size - 1)     
            mask, mask_white = random_walk(canvas, canvas_white, ini_x, ini_y, hole_size)
            mask_list.append(mask) 
            mask_white_list.append(mask_white) 
            mask_list.append(mask) 
            mask_white_list.append(mask_white) 
        
        img0 = torch.from_numpy(mask_white_list[0])
        img0 = torch.unsqueeze(img0,0)
        mask_white_all = torch.unsqueeze(img0,0)
        #mask_white_all = torch.cat([img0], 0)
        
        
        imgg0 = torch.from_numpy(mask_list[0].transpose((2, 0, 1)))
        mask_all = torch.unsqueeze(imgg0,0)
        #mask_all = torch.cat([imgg0], 0)       
  
 
        maskk = mask_white_all.type(torch.FloatTensor).cuda()
        maskk_random = mask_all.type(torch.FloatTensor).cuda()

        input_hrface = imgs_GT - imgs_GT * maskk + maskk_random  # 3*32*32  inputlr
       
       
        #######  INPUT Occluded-LR face
        input_lrface = F.interpolate(input_hrface, size=16, mode='bilinear', align_corners=True)
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
        
        
        ###### Generator
        lr_face_out_16, landmarks_en_out_16, lr_face_out_32, landmarks_en_out_32, lr_face_out_64, landmarks_en_out_64, sr_face_32, sr_face_64, sr_face_128 = Pro_UInet(input_lrface, input_heatmaps_16_7)


        ############################Local Discriminator
        print(sr_face_128.shape)
        
        heatmaps_lr = FAN_heatmap(imgs_GT)
        
        predict_heat = heatmaps_lr.cpu()
        
        predict_heat = predict_heat.detach().numpy()  
        
        preheat=get_peak_points(predict_heat)
        
        ########################################################
        #preheat1 = preheat[0]
        #print(preheat[0].shape)
        #prelandmark1=landmarks_68_to_5(preheat1)
        #prelandmark1 =prelandmark1.astype(np.uint8)
        
        #imgs_numpy0 = sr_face_128.cpu().squeeze(0).permute(1,2,0)
        #imgs_numpy = imgs_numpy0.detach().numpy()
        #imgs_numpy = (imgs_numpy * 0.5 + 0.5) * 255
        #imgs_numpy = imgs_numpy.astype(np.uint8)
        #img_genunet = Image.fromarray(imgs_numpy)
        #img_genunet = img_genunet.resize((128,128) , Image.LANCZOS)
        #batch = process(img_genunet, prelandmark1 )
        #to_tensor = transforms.ToTensor()
        #for k in batch:
            #batch[k] = to_tensor( batch[k] )
            #batch[k] = batch[k] * 2.0 - 1.0

        #batch['left_eye'] = batch['left_eye'].unsqueeze(0).cuda()
        #print(batch['left_eye'].shape)
        #batch['nose'] = batch['nose'].unsqueeze(0).cuda()
        #print(batch['nose'].shape)
        #batch['mouth'] = batch['mouth'].unsqueeze(0).cuda()
        #batch['right_eye'] = batch['right_eye'].unsqueeze(0).cuda()
        #print(out1.shape)
        ########################################################
       
        whole_face_list = []


        for j in range(opt.batch_size):
            preheat1 = preheat[j, :, :]
            print(preheat[j].shape)
            prelandmark1=landmarks_68_to_5(preheat1)
            prelandmark1 =prelandmark1.astype(np.uint8)
        
            imgs_numpy0 = imgs_GT[j, :, :, :].cpu().squeeze(0).permute(1,2,0)
            imgs_numpy = imgs_numpy0.detach().numpy()
            imgs_numpy = (imgs_numpy * 0.5 + 0.5) * 255
            imgs_numpy = imgs_numpy.astype(np.uint8)
            img_genunet = Image.fromarray(imgs_numpy)
            img_genunet = img_genunet.resize((128,128), Image.LANCZOS)
            batch = process(img_genunet, prelandmark1 )
            to_tensor = transforms.ToTensor()
            for k in batch:
                batch[k] = to_tensor( batch[k] )
                batch[k] = batch[k] * 2.0 - 1.0
                batch['whole_face'] = batch['nose']             
            nose_list.append(batch['whole_face']) 
        
        nose_list[0] = torch.unsqueeze(nose_list[0], 0)
        nose_list[1] = torch.unsqueeze(nose_list[1], 0)
        nose_list[2] = torch.unsqueeze(nose_list[2], 0)
        nose_list[3] = torch.unsqueeze(nose_list[3], 0)
        nose_all = torch.cat((nose_list[0], nose_list[1], nose_list[2], nose_list[3]), dim=0)  
        
        nose_end = nose_all.type(torch.FloatTensor).cuda()
        print(nose_end.shape)

        mse = criterion_GAN(sr_face_128.data, imgs_GT.data)
        #print(mse)
        psnr = 10 * log10(1 / mse)
        #print(psnr)
        avg_psnr = avg_psnr + psnr

        batches_done = i

        # Save image sample
        save_image(torch.cat((input_lrface.data, lr_face_out_16.data, imgs_lr16.data), -1),
                   'test_results/all/a%d.png' % batches_done, normalize=True)
        save_image(torch.cat((sr_face_32.data, lr_face_out_32.data, imgs_lr32.data), -1),
                   'test_results/all/b%d.png' % batches_done, normalize=True)
        save_image(torch.cat((sr_face_64.data, lr_face_out_64.data, imgs_lr64.data), -1),
                   'test_results/all/c%d.png' % batches_done, normalize=True)
        save_image(torch.cat((input_lrup.data, sr_face_128.data, imgs_GT.data), -1),
                   'test_results/all/d%d.png' % batches_done,
                   normalize=True)
        save_image(nose_end.data, 'test_results/mask-128/%d.png' % batches_done,
                   normalize=True)
        save_image(maskk_random.data, 'test_results/mask-128/r%d.png' % batches_done,
                   normalize=True)
        save_image(imgs_lr16.data, 'test_results/lr-16/%d.png' % batches_done,
                   normalize=True)
        save_image(imgs_lr32.data, 'test_results/lr-32/%d.png' % batches_done,
                   normalize=True)
        save_image(imgs_lr64.data, 'test_results/lr-64/%d.png' % batches_done,
                   normalize=True)
        save_image(sr_face_128.data, 'test_results/sr-128/%03d.png' % batches_done,
                   normalize=True)
        save_image(imgs_GT.data, 'test_results/GT/%d.png' % batches_done,
                   normalize=True)
        save_image(input_lrface.data, 'test_results/oc-lr-16/%d.png' % batches_done,
                   normalize=True)
        save_image(input_hrface.data, 'test_results/oc-lr-128/%d.png' % batches_done,
                   normalize=True)

    avg = avg_psnr / len(test_loader)
    avg_all = "[Epoch {:d}/{:d}][avg_psnr: {:f}]".format(epoch, 100, avg)

    with open("Test_PSNR.txt", 'a') as f:
        f.write(avg_all + "\n")


    # --------------train-------------
    for i, imgs in enumerate(train_loader):
        print("Current batch : {}".format(i))

        # 'lr32': img_lr32, 'lr64': img_lr64, 'hr128': img_hr128, 'hr256': img_hr256
        ## Data preprocess
        imgs_lr16 = imgs['lr16'].cuda()
        imgs_lr32 = imgs['lr32'].cuda()
        imgs_lr64 = imgs['lr64'].cuda()
        imgs_hr128 = imgs['hr128'].cuda()
        imgs_GT = imgs['hr128'].cuda()
        imgs_256 = imgs['hr256'].cuda()
        

        mask_list = []
        mask_white_list = []
        for j in range(opt.batch_size):   
            canvas = np.zeros((opt.image_size, opt.image_size, 3))
            canvas_white = np.zeros((opt.image_size, opt.image_size))
            ini_x = random.randint(0, opt.image_size - 1)
            ini_y = random.randint(0, opt.image_size - 1)     
            action_list = [[0, 1], [0, -1], [1, 0], [-1, 0]]
            hole_size=((64, 72), (64, 72))
            mask, mask_white = random_walk(canvas, canvas_white, ini_x, ini_y, hole_size)
            mask_list.append(mask) 
            mask_white_list.append(mask_white) 
        
        img3 = torch.from_numpy(mask_white_list[3])
        img3 = torch.unsqueeze(img3,0)
        img3 = torch.unsqueeze(img3,0)
        img2 = torch.from_numpy(mask_white_list[2])
        img2 = torch.unsqueeze(img2,0)
        img2 = torch.unsqueeze(img2,0)
        img1 = torch.from_numpy(mask_white_list[1])
        img1 = torch.unsqueeze(img1,0)
        img1 = torch.unsqueeze(img1,0)
        img0 = torch.from_numpy(mask_white_list[0])
        img0 = torch.unsqueeze(img0,0)
        img0 = torch.unsqueeze(img0,0)
        mask_white_all = torch.cat([img0, img1, img2, img3], 0)
        
        
        imgg3 = torch.from_numpy(mask_list[3].transpose((2, 0, 1)))
        imgg3 = torch.unsqueeze(imgg3,0)
        imgg2 = torch.from_numpy(mask_list[2].transpose((2, 0, 1)))
        imgg2 = torch.unsqueeze(imgg2,0)
        imgg1 = torch.from_numpy(mask_list[1].transpose((2, 0, 1)))
        imgg1 = torch.unsqueeze(imgg1,0)
        imgg0 = torch.from_numpy(mask_list[0].transpose((2, 0, 1)))
        imgg0 = torch.unsqueeze(imgg0,0)
        mask_all = torch.cat([imgg0, imgg1, imgg2, imgg3], 0)
        
  
        maskk = mask_white_all.type(torch.FloatTensor).cuda()
        maskk_random = mask_all.type(torch.FloatTensor).cuda()

        input_hrface = imgs_GT - imgs_GT * maskk + maskk_random  # 3*32*32  inputlr
       
       
        #######  INPUT Occluded-LR face
        input_lrface = F.interpolate(input_hrface, size=16, mode='bilinear', align_corners=True)
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


        #######  GT  Heatmaps
        heatmaps_64_GT = FAN_heatmap(imgs_256)
        heatmaps_16_GT = F.interpolate(heatmaps_64_GT, size=16, mode='bilinear', align_corners=True)
        heatmaps_32_GT = F.interpolate(heatmaps_64_GT, size=32, mode='bilinear', align_corners=True)

        # GT 7 key heatmaps
        left_eye1 = heatmaps_16_GT[:, 36:37, :, :]
        left_eye2 = heatmaps_16_GT[:, 39:40, :, :]
        right_eye1 = heatmaps_16_GT[:, 42:43, :, :]
        right_eye2 = heatmaps_16_GT[:, 45:46, :, :]
        nose = heatmaps_16_GT[:, 30:31, :, :]
        mouth1 = heatmaps_16_GT[:, 48:49, :, :]
        mouth2 = heatmaps_16_GT[:, 54:55, :, :]

        # GT 7 key heatmaps
        key_heatmaps_16_7 = torch.cat([left_eye1, left_eye2, right_eye1, right_eye2, nose, mouth1, mouth2], 1)

        # GT 42 key heatmaps
        gl_32 = heatmaps_32_GT[:, 0:17, :, :]
        mei_32_1 = heatmaps_32_GT[:, 17:22, :, :]
        mei_32_2 = heatmaps_32_GT[:, 22:27, :, :]
        left_eye_32_1 = heatmaps_32_GT[:, 36:37, :, :]
        left_eye_32_2 = heatmaps_32_GT[:, 39:40, :, :]
        right_eye_32_1 = heatmaps_32_GT[:, 42:43, :, :]
        right_eye_32_2 = heatmaps_32_GT[:, 45:46, :, :]
        nose_32 = heatmaps_32_GT[:, 27:31, :, :]
        mouth_32_1 = heatmaps_32_GT[:, 48:49, :, :]
        mouth_32_2 = heatmaps_32_GT[:, 54:55, :, :]
        mouth_32 = heatmaps_32_GT[:, 60:65, :, :]

        # GT 42 key heatmaps
        key_heatmaps_32_42 = torch.cat(
            [left_eye_32_1, left_eye_32_2, right_eye_32_1, right_eye_32_2, nose_32, mouth_32_1, mouth_32_2, gl_32,
             mei_32_1, mei_32_2, mouth_32], 1)

       
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((imgs_GT.size(0), *Global_D.output_shape))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_GT.size(0), *Global_D.output_shape))), requires_grad=False)
        
        
        ###### Generator
        lr_face_out_16, landmarks_en_out_16, lr_face_out_32, landmarks_en_out_32, lr_face_out_64, landmarks_en_out_64, sr_face_32, sr_face_64, sr_face_128 = Pro_UInet(input_lrface, input_heatmaps_16_7)


        ##### UInet3
        loss_style_net3, loss_smoth_net3 = Style_loss(sr_face_128, imgs_GT)

        # l2 loss
        loss_l2_net3_64 = criterion_GAN(lr_face_out_64, imgs_lr64)
        loss_l2_net3_128 = criterion_GAN(sr_face_128, imgs_GT)

        # content loss l1 loss
        gt_features_32 = Variable(feature_extractor(imgs_lr32).data, requires_grad=False)
        gt_features_64 = Variable(feature_extractor(imgs_lr64).data, requires_grad=False)
        gt_features_128 = Variable(feature_extractor(imgs_GT).data, requires_grad=False)
        sr_features_64_2 = feature_extractor(lr_face_out_64)
        sr_features_128 = feature_extractor(sr_face_128)
        loss_content_64_2 = criterion_GAN(gt_features_64, sr_features_64_2)
        loss_content_128 = criterion_GAN(gt_features_128, sr_features_128)

        # landmark loss
        loss_heatmap_net3 = criterion_GAN(landmarks_en_out_64, heatmaps_64_GT)

        # Adversarial loss_128
        gen_validity_global = Global_D(sr_face_128)
        loss_GAN_global = criterion_GAN(gen_validity_global, valid)
        
        loss_GAN = loss_GAN_global 

        lossG_net3 = loss_style_net3 + (1e-2) * loss_smoth_net3 + loss_l2_net3_64 + loss_l2_net3_128 + (
            1e-2) * loss_content_64_2 + (1e-2) * loss_content_128 + loss_heatmap_net3 + (1e-3) * loss_GAN


        ##### UInet2
        loss_style_net2, loss_smoth_net2 = Style_loss(sr_face_64, imgs_lr64)
        loss_l2_net2_32 = criterion_GAN(lr_face_out_32, imgs_lr32)
        loss_l2_net2_64 = criterion_GAN(sr_face_64, imgs_lr64)

        gt_features_32 = Variable(feature_extractor(imgs_lr32).data, requires_grad=False)
        gt_features_64 = Variable(feature_extractor(imgs_lr64).data, requires_grad=False)
        sr_features_32_2 = feature_extractor(lr_face_out_32)
        sr_features_64_1 = feature_extractor(sr_face_64)
        loss_content_32_2 = criterion_GAN(gt_features_32, sr_features_32_2)
        loss_content_64_1 = criterion_GAN(gt_features_64, sr_features_64_1)

        loss_heatmap_net2 = criterion_GAN(landmarks_en_out_32, key_heatmaps_32_42)

        lossG_net2 = 10 * loss_style_net2 + (1e-1) * loss_smoth_net2 + loss_l2_net2_32 + loss_l2_net2_64 + (
            1e-2) * loss_content_32_2 + (1e-2) * loss_content_64_1 + loss_heatmap_net2
     

        ##### UInet1
        loss_style_net1, loss_smoth_net1 = Style_loss(sr_face_32, imgs_lr32)

        loss_l2_net1_16 = criterion_GAN(lr_face_out_16, imgs_lr16)
        loss_l2_net1_32 = criterion_GAN(sr_face_32, imgs_lr32)

        gt_features_16 = Variable(feature_extractor(imgs_lr16).data, requires_grad=False)
        gt_features_32 = Variable(feature_extractor(imgs_lr32).data, requires_grad=False)
        sr_features_16 = feature_extractor(lr_face_out_16)
        sr_features_32_1 = feature_extractor(sr_face_32)
        loss_content_16 = criterion_GAN(gt_features_16, sr_features_16)
        loss_content_32_1 = criterion_GAN(gt_features_32, sr_features_32_1)

        

        loss_heatmap_net1 = criterion_GAN(landmarks_en_out_16, key_heatmaps_16_7)

        lossG_net1 = 10 * loss_style_net1 + (1e-1) * loss_smoth_net1 + 10 * loss_l2_net1_16 + 10 * loss_l2_net1_32 + (1e-1) * loss_content_16 + (
                         1e-1) * loss_content_32_1 + loss_heatmap_net1
                         
        loss_all = lossG_net1 + lossG_net2 + lossG_net3

        optimizer_G.zero_grad()
        loss_all.backward(retain_graph=True)
        optimizer_G.step()
        
   
        G_Loss = "[Epoch {:d}/{:d}] [Batch {:d}/{:d}] [lossG_net1: {:f}][lossG_net2: {:f}] [lossG_net3: {:f}] [lossD: {:f}]".format(
            epoch, 100, i, len(train_loader), lossG_net1.item(), lossG_net2.item(), lossG_net3.item(), loss_GAN.item())
        print(G_Loss)

        # Save log
        with open("lossG.txt", 'a') as f:
            f.write(G_Loss + "\n")

        batches_done = epoch * len(train_loader) + i

        if batches_done % opt.sample_interval == 0:
            # Save image sample
            save_image(torch.cat((input_lrface.data, lr_face_out_16.data, imgs_lr16.data), -1),
                       'images/a%d.png' % batches_done, normalize=True)
            save_image(torch.cat((sr_face_32.data, lr_face_out_32.data, imgs_lr32.data), -1),
                       'images/b%d.png' % batches_done, normalize=True)
            save_image(torch.cat((sr_face_64.data, lr_face_out_64.data, imgs_lr64.data), -1),
                       'images/c%d.png' % batches_done, normalize=True)
            save_image(torch.cat((input_lrup.data, input_hrface, sr_face_128.data, imgs_GT.data), -1),
                       'images/d%d.png' % batches_done, normalize=True)

            #writer.add_scalar("lossG1", lossG_net1.item(), batches_done)
            #writer.add_scalar("lossG2", lossG_net2.item(), batches_done)
            #writer.add_scalar("lossG3", lossG_net3.item(), batches_done)
            #writer.add_scalar("G_adv_global", loss_GAN_global.item(), batches_done)
            #writer.add_scalar("G_adv_local", loss_GAN_local.item(), batches_done)
            #writer.add_image("SR_result", sr_face_128.data, batches_done)

        if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
            torch.save(Pro_UInet.state_dict(), 'saved_models/Generator%d.pth' % epoch)
            

