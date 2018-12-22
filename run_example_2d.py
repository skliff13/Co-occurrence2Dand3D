import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2gray

from cooccur2D import cooccur2D

def example2d():

    im = rgb2gray(io.imread('test_data/lena256.jpg')).astype(float)
    comatrix = cooccur2D(im, i_bins=6, g_bins=3, a_bins=6, dists=(1, 3))
    print('Matrix shape: (Num_Distances, A_bins, G_bins, G_bins, I_bins, I_bins)')
    print(comatrix.shape)

    tests = [['lena256.jpg', None, 6, 1, 1],
             ['lena256.jpg', None, 1, 6, 1],
             ['lena256.jpg', 'roi1.png', 1, 6, 1],
             ['lena256.jpg', 'roi2.png', 1, 6, 1],
             ['noise.png', None, 1, 6, 1],
             ['grass2.png', None, 1, 6, 1],
             ['grass2.png', None, 1, 6, 10]]

    for test in tests:
        image_file = test[0]
        mask_file = test[1]
        intensity_bins = test[2]
        angle_bins = test[3]
        distance = test[4]

        im = rgb2gray(io.imread('test_data/' + image_file)).astype(float)
        if mask_file:
            mask = rgb2gray(io.imread('test_data/' + mask_file)) > 0
        else:
            mask = None

        comatrix = cooccur2D(im, i_bins=intensity_bins, a_bins=angle_bins, dists=(distance,), mask=mask)

        plt.subplots(1, 3, figsize=(15, 4.5))
        ax = plt.subplot(131)
        ax.imshow(im, cmap='gray')
        ax.set_title('Image: ' + image_file, va='bottom')

        ax = plt.subplot(132)
        if mask is not None:
            ax.imshow(im * mask, cmap='gray')
        else:
            ax.imshow(im, cmap='gray')
        ax.set_title('Image + ROI', va='bottom')

        ax = plt.subplot(133)

        if len(comatrix.shape) == 2:
            ax.imshow(comatrix, cmap='jet')
        elif len(comatrix.shape) == 1:
            plt.bar(np.arange(comatrix.shape[0]), comatrix / np.max(comatrix))
        ax.set_title('Output (I_bins=%i, A_bins=%i)' % (intensity_bins, angle_bins), va='bottom')

        plt.show()


if __name__ == '__main__':
    example2d()