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

def dataloader(filepath):

  left_fold  = 'colored_0/'
  right_fold = 'colored_1/'
  disp_noc   = 'disp_occ/'

#   image = [img for img in os.listdir(filepath+left_fold)]
  image = [str(i) for i in range(1, 8000)]
  random.shuffle(image)

  train = image[:]
  val   = image[7500:]

#   left_train = np.array([i for i in os.listdir(filepath) if i.find('carla-left-')>-1])[train].tolist()
#   right_train = np.array([i for i in os.listdir(filepath) if i.find('carla-right-')>-1])[train].tolist()
#   disp_train = np.array([i for i in os.listdir(filepath) if i.find('carla-depth-')>-1])[train].tolist()

#   left_val = np.array([i for i in os.listdir(filepath) if i.find('carla-left-')>-1])[train].tolist()
#   right_val = np.array([i for i in os.listdir(filepath) if i.find('carla-right-')>-1])[train].tolist()
#   disp_val = np.array([i for i in os.listdir(filepath) if i.find('carla-depth-')>-1])[train].tolist()

  left_train  = [filepath+'carla-left-' +img+'00.png' for img in train]
  right_train = [filepath+'carla-right-'+img+'00.png' for img in train]
  disp_train  = [filepath+'carla-depth-'+img+'00.pkl' for img in train]

  left_val  = [filepath+'carla-left-' +img+'00.png' for img in val]
  right_val = [filepath+'carla-right-'+img+'00.png' for img in val]
  disp_val  = [filepath+'carla-depth-'+img+'00.pkl' for img in val]

  return left_train, right_train, disp_train, left_val, right_val, disp_val
