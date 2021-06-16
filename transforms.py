from torch.utils.data import Dataset
import torch
import numpy as np
import random
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import normalize

# 加一微小的噪声
class Mask:
    
    def __init__(self, mask_scale = 0.1):
        self.mask_scale = mask_scale
        
    def __call__(self,image,label):
        self.mask = np.random.rand(*image.shape) * self.mask_scale
        
        # max_test=np.max(label)
        return image + self.mask,label

# 除去极端数据    
class Window:

    def __init__(self, window_min, window_max):
        self.window_min = window_min
        self.window_max = window_max

    def __call__(self, image,label=None):
        image = np.clip(image, self.window_min, self.window_max)
        
        # max_test=np.max(label)
        return image,label

# 数据归一化
class MinMaxNorm:

    def __init__(self, low, high):
        self.low = low
        self.high = high

    def __call__(self, image, label=None):
        image = (image - self.low) / (self.high - self.low)
        image = image * 2 - 1
        
        # max_test=np.max(label)
        return image,label

# 调整大小，这个不需要    
class Resize:
    def __init__(self, scale):
        # self.shape = [shape, shape, shape] if isinstance(shape, int) else shape
        self.scale = scale

    def __call__(self, image, label):
        image, label = image.unsqueeze(0), label.unsqueeze(0)
        img = F.interpolate(image, scale_factor=(1,self.scale,self.scale),mode='trilinear', align_corners=False, recompute_scale_factor=True)
        label = F.interpolate(label, scale_factor=(1,self.scale,self.scale), mode="nearest", recompute_scale_factor=True)
        
        max_test=torch.max(label)
        return img[0],label[0]

class RandomResize:
    def __init__(self,s_rank, w_rank,h_rank):
        self.w_rank = w_rank
        self.h_rank = h_rank
        self.s_rank = s_rank

    def __call__(self, image,label):
        random_s = random.randint(int(image.shape[0]*self.w_rank[0]),int(image.shape[0]*self.w_rank[1]))
        random_h = random.randint(int(image.shape[1]*self.w_rank[0]),int(image.shape[1]*self.w_rank[1]))
        random_w = random.randint(int(image.shape[2]*self.w_rank[0]),int(image.shape[2]*self.w_rank[1]))
        self.shape = [random_s,random_h,random_w]
        image,label = torch.from_numpy(image).unsqueeze(0).unsqueeze(0),torch.from_numpy(label).unsqueeze(0).unsqueeze(0)
        # print(image.shape,image.dtype,label.shape,label.dtype)
        image = F.interpolate(image, size=self.shape,mode='trilinear', align_corners=False)
        label = F.interpolate(label.float(), size=self.shape, mode="nearest")
        image_arr=image[0][0].numpy()
        label_arr=np.uint8(label[0][0].numpy())
        return image_arr,label_arr

# 随机缩放
class RandomCrop:
    def __init__(self, slices):
        self.slices =  slices

    def _get_range(self, slices, crop_slices):
        if slices < crop_slices:
            start = 0
        else:
            start = random.randint(0, slices - crop_slices)
        end = start + crop_slices
        if end > slices:
            end = slices
        return start, end

    def __call__(self, img, label):

        ss, es = self._get_range(img.shape[1], self.slices)
        
        # print(self.shape, img.shape, mask.shape)
        tmp_img = np.zeros((img.shape[0], self.slices, img.shape[2]))
        tmp_label = np.zeros((label.size(0), self.slices, label.size(2)))
        tmp_img[:,:es-ss] = img[:,ss:es]
        tmp_label[:,:es-ss] = label[:,ss:es]
        
        # max_test=torch.max(tmp_label)
        return tmp_img,tmp_label

# 左右翻转
class RandomFlip_LR:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob <= self.prob:
            img = np.flip(img,1)
        return img

    def __call__(self, image,label):
        prob = random.uniform(0, 1)
        image=self._flip(image, prob)
        label=self._flip(label, prob)
        # max_test=torch.max(label)
        return image,label

# 上下翻转
class RandomFlip_UD:
    def __init__(self, prob=0.5):
        self.prob = prob

    def _flip(self, img, prob):
        if prob <= self.prob:
            img = np.flip(img,2)
        return img

    def __call__(self, image,label):
        prob = random.uniform(0, 1)
        image=self._flip(image, prob)
        label=self._flip(label, prob)
        # max_test=torch.max(label)
        return image,label

# 随机旋转
class RandomRotate:
    def __init__(self, max_cnt=3):
        self.max_cnt = max_cnt

    def _rotate(self, img, cnt):
        img = torch.rot90(img,cnt,[1,2])
        return img

    def __call__(self, img):
        cnt = random.randint(0,self.max_cnt)
        return self._rotate(img, cnt)#, self._rotate(mask, cnt)


class Center_Crop:
    def __init__(self, base, max_size):
        self.base = base  # base默认取16，因为4次下采样后为1
        self.max_size = max_size 
        if self.max_size%self.base:
            self.max_size = self.max_size - self.max_size%self.base # max_size为限制最大采样slices数，防止显存溢出，同时也应为16的倍数
    def __call__(self, img , label):
        if img.size(1) < self.base:
            return None
        slice_num = img.size(1) - img.size(1) % self.base
        slice_num = min(self.max_size, slice_num)

        left = img.size(1)//2 - slice_num//2
        right =  img.size(1)//2 + slice_num//2

        crop_img = img[:,left:right]
        crop_label = label[:,left:right]
        return crop_img, crop_label

class ToTensor:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()

    def __call__(self, image,label=None):
        # max_test=np.max(label)
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)
        # max_test=torch.max(label)
        return image, label


# class Normalize:
#     def __init__(self, mean, std):
#         self.mean = mean
#         self.std = std

#     def __call__(self, img):
#         return normalize(img, self.mean, self.std, False)#, mask

# 原本是为
# class Compose:
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, img, mask):
#         for t in self.transforms:
#             img, mask = t(img, mask)
#         return img, mask