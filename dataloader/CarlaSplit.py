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

    f = []
    f.append(('town01_start1/', '015', 626))
    f.append(('town01_start10/', '035', 635))
    f.append(('town01_start20/', '010', 640))
    f.append(('town01_start30/', '029', 623))
    f.append(('town02_start1/', '613', 656))
    f.append(('town02_start10/', '663', 662))
    f.append(('town02_start20/', '622', 603))
    f.append(('town02_start30/', '645', 285))
    f.append(('town03_start1/', '387', 660))
    f.append(('town03_start10/', '346', 632))
    f.append(('town03_start20/', '359', 673))
    f.append(('town04_start1/', '967', 647))
    f.append(('town04_start10/', '939', 635))
    f.append(('town04_start20/', '892', 641))
    f.append(('town05_start1/', '976', 600))
    f.append(('town05_start10/', '008', 600))
    f.append(('town05_start20/', '959', 604))

    left_train, right_train, disp_train, left_val, right_val, disp_val = [], [], [], [], [], []
    
    for folder, code, num in f:
        image = [str(i) for i in range(10, num+1)]
        random.shuffle(image)

        left_train  += [filepath+folder+'carla-center-'+img+code+'.png' for img in image]
        right_train += [filepath+folder+'carla-right-' +img+code+'.png' for img in image]
        disp_train  += [filepath+folder+'carla-depth-' +img+code+'.pkl' for img in image]

    left_val  += [filepath+'town02_start30/carla-center-'+str(img)+'645.png' for img in range(10, 286)]
    right_val += [filepath+'town02_start30/carla-right-' +str(img)+'645.png' for img in range(10, 286)]
    disp_val  += [filepath+'town02_start30/carla-depth-' +str(img)+'645.pkl' for img in range(10, 286)]

    return left_train, right_train, disp_train, left_val, right_val, disp_val
