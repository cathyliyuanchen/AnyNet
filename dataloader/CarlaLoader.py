import os
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import random
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
from . import preprocess

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def default_loader(path):
    return Image.open(path).convert('RGB')

def disparity_loader(path):
    return pd.read_pickle(path)


class myImageFloder(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader= disparity_loader):
 
        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left  = self.left[index]
        right = self.right[index]
        disp_L= self.disp_L[index]

        left_img = self.loader(left)
        right_img = self.loader(right)
        dataL = self.dploader(disp_L)
        # mask = np.logical_and(dataL < 0.1*(np.ones((600, 800))), dataL > 0.002*(np.ones((600, 800)))).astype(int)
#         mask = (dataL < 0.1*(np.ones((600, 800)))).astype(int)
#         dataL = (1/3)*np.reciprocal(dataL)*mask
        dataL = (0.5 * 0.5 * 1920) * np.reciprocal(1000 * dataL) # camera_dist = 0.5m, focal_len = 0.5m, width = 1920
        dataL = np.ascontiguousarray(dataL,dtype=np.float32)
        
        #not cropping
        left_img = left_img.crop((0, 360, 1920, 900))
        right_img = right_img.crop((0, 360, 1920, 900))
        dataL = dataL[360:900]
        processed = preprocess.get_transform(augment=False)  
        left_img       = processed(left_img)
        right_img      = processed(right_img)
        
        return left_img, right_img, dataL
        
        if self.training:  

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))
            
            w, h = left_img.size
            th, tw = 256, 512

            x1 = random.randint(0, w - tw)
            y1 = random.randint(0, h - th)

            left_img = left_img.crop((x1, y1, x1 + tw, y1 + th))
            right_img = right_img.crop((x1, y1, x1 + tw, y1 + th))

            dataL = dataL[y1:y1 + th, x1:x1 + tw]

            processed = preprocess.get_transform(augment=False)  
            left_img   = processed(left_img)
            right_img  = processed(right_img)

            return left_img, right_img, dataL
        else:
            w, h = left_img.size

            left_img = left_img.crop((w-800, h-600, w, h))
            right_img = right_img.crop((w-800, h-600, w, h))
            w1, h1 = left_img.size
            # if uncomment the line that crops the output, 600 should be 596
            dataL = dataL.crop((w-800, h-600, w, h))    

            processed = preprocess.get_transform(augment=False)  
            left_img       = processed(left_img)
            right_img      = processed(right_img)

            return left_img, right_img, dataL

    def __len__(self):
        return len(self.left)
