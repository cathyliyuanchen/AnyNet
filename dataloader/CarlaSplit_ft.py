import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
import random

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataloader(filepath, log, args=False):

    left_fold  = 'colored_0/'
    right_fold = 'colored_1/'
    disp_noc   = 'disp_occ/'

    #   image = [img for img in os.listdir(filepath+left_fold)]
    image = [str(i) for i in range(1, 2700)]
    random.shuffle(image)

    train = image[:2500]
    val   = image[2500:]
    log.info(val)

    left_train  = [filepath+'carla-left-' +img+'.png' for img in train]
    right_train = [filepath+'carla-center-'+img+'.png' for img in train]
    disp_train  = [filepath+'carla-depth-'+img+'.pkl' for img in train]


    left_val  = [filepath+'carla-left-' +img+'.png' for img in val]
    right_val = [filepath+'carla-center-'+img+'.png' for img in val]
    disp_val  = [filepath+'carla-depth-'+img+'.pkl' for img in val]

    return left_train, right_train, disp_train, left_val, right_val, disp_val
