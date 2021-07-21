#################################################################
## CREDITS: modified from
## https://www.programmersought.com/article/21535487135/ and
## https://blog.csdn.net/weixin_44899143/article/details/89186891
#################################################################

import numpy as np
from matplotlib import pyplot as plt
import os
import re


def read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$',
                             pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<'  # littel endian
            scale = -scale
        else:
            endian = '>'  # big endian

        disparity = np.fromfile(pfm_file, endian + 'f')
    #
    img = np.reshape(disparity, newshape=(height, width, channels))
    img = np.flipud(img).astype('uint8')
    #
    # plt.show(img, "disparity")
    png_file_path = pfm_file_path[:-4] + ".png"
    # plt.imsave(os.path.join(png_file_path), img)

    # return disparity, [(height, width, channels), scale]
    return img


if __name__ == "__main__":
    basename = 'Classroom1-perfect'

    path = 'images/' + basename + '/disp0.pfm'
    img = read_pfm(path)
    # print(type(img))
    # print(img.shape)
    plt.imshow(img)
    plt.show()