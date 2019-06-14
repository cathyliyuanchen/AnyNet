import numpy as np
from sklearn.decomposition import PCA
import cv2

def main():

    image = [str(10*i) for i in range(1, 200)]
    filepath = './dataset/'
    left_train  = [filepath+'colored_0/carla-left-' +img+'00.png' for img in image]
    right_train = [filepath+'colored_1/carla-right-'+img+'00.png' for img in image]

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

    return 



if __name__ == '__main__':
    main()
