import numpy as np
from sklearn.decomposition import PCA
import cv2
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

def main():

    image = [str(10*i) for i in range(1, 200)]
    filepath = '/mnt/erdos_data1/carla_0.9.5_data/autopilot-town1-2-3/'
    left_train  = [filepath+'carla-left-' +img+'.png' for img in image]
    right_train = [filepath+'carla-center-'+img+'.png' for img in image]

    data = np.array([cv2.imread(i) for i in left_train+right_train])   #BGR
    data = data.reshape(-1,3)/255
    data[:, 0], data[:, 2] = data[:, 2], data[:, 0].copy()   #RGB

    print('mean')
    print(np.mean(data, axis=0))
    print('std')
    print(np.std(data, axis=0))

    pca = PCA()
    pca.fit(data)
    print(pca.components_)
    print(pca.explained_variance_)

#     img_l = mpimg.imread("./carla-left-200.png")
#     img_r = mpimg.imread("./carla-right-200.png")
#     img = np.concatenate((img_l, img_r), axis=1)
#     plt.imshow(img)
#     plt.show()

    return 



if __name__ == '__main__':
    main()
