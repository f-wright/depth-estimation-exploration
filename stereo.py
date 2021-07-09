import numpy as np
import cv2 
from matplotlib import pyplot as plt
from matplotlib import cm

# fx = 7190.247               # lense focal length
# baseline = 174.945          # distance in mm between the two cameras (values from middlebury)
# units = 0.001               # depth units
# doffs=342.523               # x-difference of principal points, following https://vision.middlebury.edu/stereo/data/scenes2014/#description

# texture_threshold = 2000      # 10 by default


# imgL = cv2.imread('images/im0_left.png', cv2.IMREAD_GRAYSCALE)
# imgR = cv2.imread('images/im0_right.png', cv2.IMREAD_GRAYSCALE)

# Classroom values
doffs=113.186
baseline=237.604
fx = 3920.793
doffs=113.186

disparities=0
block=15

imgL = cv2.imread('images/Classroom1-imperfect/im0.png', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('images/Classroom1-imperfect/im1.png', cv2.IMREAD_GRAYSCALE)

plt.imshow(imgL, cmap="gray")
plt.imshow(imgR, cmap="gray")
plt.show()

sbm = cv2.StereoBM_create(numDisparities=disparities,blockSize=block)
# sbm.setTextureThreshold(texture_threshold)


# calculate disparities
disparity = sbm.compute(imgL, imgR)
print(disparity)
# show disparity
plt.imshow(disparity, 'gray')
plt.show()




depth = np.zeros(shape=imgL.shape).astype(float)
depth[disparity > 0] = (fx * baseline) / (doffs * disparity[disparity > 0])
#print(depth)

plt.imshow(depth)
plt.show()