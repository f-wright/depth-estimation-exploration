import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pfm_conversion import read_pfm

basename = 'Backpack-perfect'

imgL = cv.imread('images/' + basename + '/im0.png', 0)  # type Mat
imgR = cv.imread('images/' + basename + '/im1.png', 0)

##############################################
##### setup stereoBM
##############################################

sbmParams = {
    'SWS': 5,  #SADWindowSize
    'PFS': 5,  #PreFilterSize
    'PFC': 29,  #PreFiltCap
    'MDS': 50,  #MinDisparity
    'NOD': 128,  #NumDisparities
    'TTH': 100,  #TxtrThrshld
    'UR': 10,  #UniquenessRatio
    'SR': 15,  #SpklRng
    'SPWS': 100,  #SpklWinSize
}

sbm = cv.StereoBM_create(numDisparities=16, blockSize=15)
sbm.setPreFilterType(1)
sbm.setPreFilterSize(sbmParams['PFS'])
sbm.setPreFilterCap(sbmParams['PFC'])
sbm.setMinDisparity(sbmParams['MDS'])
sbm.setNumDisparities(sbmParams['NOD'])
sbm.setTextureThreshold(sbmParams['TTH'])
sbm.setUniquenessRatio(sbmParams['UR'])
sbm.setSpeckleRange(sbmParams['SR'])
sbm.setSpeckleWindowSize(sbmParams['SPWS'])

##############################################
##### get calibration info automatically
##############################################

calibPath = 'images/' + basename + '/calib.txt'
calibData = {}
with open(calibPath) as f:
    for line in f:
        (key, val) = line.strip().split('=')
        calibData[key] = val
print(calibData)
fx = float(calibData['cam0'][1:].split()[0])
baseline = float(calibData['baseline'])
doffs = float(calibData['doffs'])
print('fx: ', fx)
print('baseline: ', baseline)
print('doffs: ', doffs)

##############################################
##############################################

disparity = sbm.compute(imgL, imgR)

depth = np.zeros(shape=imgL.shape).astype(float)  # initialize np
depth[disparity > 0] = (fx * baseline) / (doffs + disparity[disparity > 0]
                                          )  # populate np
# print(depth)

##############################################
##############################################

disp0 = read_pfm('images/' + basename + '/disp0.pfm')
depth0 = np.zeros(shape=disp0.shape).astype(float)  # initialize np
depth0[disp0 > 0] = (fx * baseline) / (doffs + disp0[disp0 > 0])  # populate np

fig, axs = plt.subplots(2, 3)
fig.suptitle(str(sbmParams))

axs[0, 0].imshow(imgL, 'gray')
estDisp = axs[0, 1].imshow(disparity, aspect='equal', cmap='viridis')
axs[0, 1].set_title("estDisp")
estDepth = axs[0, 2].imshow(depth, aspect='equal', cmap='plasma')
axs[0, 2].set_title("estDepth")
fig.colorbar(estDisp, ax=axs[0, 1])
fig.colorbar(estDepth, ax=axs[0, 2])

axs[1, 0].imshow(imgL, 'gray')
trueDisp = axs[1, 1].imshow(disp0, aspect='equal', cmap='viridis')
axs[1, 1].set_title("trueDisp")
trueDepth = axs[1, 2].imshow(depth0, aspect='equal', cmap='plasma')
axs[1, 2].set_title("trueDepth")
fig.colorbar(trueDisp, ax=axs[1, 1])
fig.colorbar(trueDepth, ax=axs[1, 2])

plt.show()

##############################################

# f = open("classroom_depth.txt", 'w')
# for row in depth:
#     for el in row:
#         f.write(str(el) + " ")
#     f.write("\n")
# f.close()