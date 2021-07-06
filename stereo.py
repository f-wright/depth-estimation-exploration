import numpy as np
import cv2 
from matplotlib import pyplot as plt
from matplotlib import cm

fx = 7190.247       # lense focal length
baseline = 174.945  # distance in mm between the two cameras (values from middlebury)
disparities = 16    # num of disparities to consider
block = 5           # block size to match
units = 0.001       # depth units
doffs=342.523       # x-difference of principal points, following https://vision.middlebury.edu/stereo/data/scenes2014/#description


imgL = cv2.imread('im0.png', 0)
imgR = cv2.imread('im1.png', 0)


sbm = cv2.StereoBM_create(numDisparities=disparities,blockSize=block)

# calculate disparities
disparity = sbm.compute(imgL, imgR)
print(disparity)
# # show disparity
# plt.imshow(disparity, 'gray')
# plt.show()


depth = np.zeros(shape=imgL.shape).astype(float)
depth[disparity > 0] = (fx * baseline) / (doffs * disparity[disparity > 0])
#print(depth)

plt.imshow(depth)
plt.show()