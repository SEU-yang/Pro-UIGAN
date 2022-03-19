import glob
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


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
    

def random_walk(canvas, canvas_white, ini_x, ini_y, hole_size):
    x = ini_x
    y = ini_y
    hole_area_crop = gen_hole_area((64, 64), (128, 128))
    hole_area = hole_area_crop
    harea_xmin, harea_ymin = hole_area[0]
    harea_w, harea_h = hole_area[1]
    hole_w = random.randint(hole_size[0][0], hole_size[0][1])
    hole_h = random.randint(hole_size[1][0], hole_size[1][1]) 
    #print(hole_w, hole_h)
    offset_x = random.randint(harea_xmin, harea_xmin + harea_w - hole_w)
    offset_y = random.randint(harea_ymin, harea_ymin + harea_h - hole_h)    
    #print(offset_x, offset_y)
    x_list = []
    y_list = []
    length = 48 * 48
    for i in range(length):
        r = random.randint(0, len(action_list) - 1)
        x = np.clip(x + action_list[r][0], a_min=offset_x, a_max=offset_x + hole_w)
        #print(x, offset_x, offset_x + hole_w)
        y = np.clip(y + action_list[r][1], a_min=offset_y, a_max=offset_y + hole_h)
        x_list.append(x)
        y_list.append(y)
    for j in range(length):    
        canvas[np.array(x_list[j]), np.array(y_list[j]), 0]=np.random.rand(1)
        canvas[np.array(x_list[j]), np.array(y_list[j]), 1]=np.random.rand(1)
        canvas[np.array(x_list[j]), np.array(y_list[j]), 2]=np.random.rand(1)
    canvas_white[np.array(x_list), np.array(y_list)]=1 
    return canvas, canvas_white
    

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

