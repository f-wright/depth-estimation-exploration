import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from pfm_conversion import read_pfm
import imageio

basename = 'Bicycle1-imperfect'

imgL = cv.imread('images/' + basename + '/im0.png', 0)  # type Mat
imgR = cv.imread('images/' + basename + '/im1.png', 0)

##############################################
##### setup stereoBM
##############################################

sbmParams = {
    'SWS': 5,  #SADWindowSize
    'PFS': 5,  #PreFilterSize
    'PFC': 29,  #PreFiltCap
    'MDS': 0,  #MinDisparity
    'NOD': 240,  #NumDisparities
    'TTH': 100,  #TxtrThrshld
    'UR': 2,  #UniquenessRatio
    'SR': 15,  #SpklRng
    'SPWS': 30,  #SpklWinSize
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

estDisp = sbm.compute(imgL, imgR) / 16

estDepth = np.zeros(shape=imgL.shape).astype(float)  # initialize np
estDepth[estDisp > 0] = (fx * baseline) / (doffs + estDisp[estDisp > 0]
                                           )  # populate np
# print(depth)

##############################################
##############################################

trueDisp = read_pfm('images/' + basename + '/disp0.pfm')
# disp0 = imageio.imread('images/' + basename + '/disp0.pfm')
trueDisp = np.asarray(trueDisp)
trueDepth = np.zeros(shape=trueDisp.shape).astype(float)  # initialize np
trueDepth[trueDisp > 0] = (fx * baseline) / (doffs + trueDisp[trueDisp > 0]
                                             )  # populate np

fig, axs = plt.subplots(2, 3)
fig.suptitle(str(sbmParams))

axs[0, 0].imshow(imgL, 'gray')
axs[0, 0].axis('off')
estDispAx = axs[0, 1].imshow(estDisp, aspect='equal', cmap='viridis')
axs[0, 1].set_title("estDisp")
axs[0, 1].axis('off')
estDepthAx = axs[0, 2].imshow(estDepth, aspect='equal', cmap='plasma')
axs[0, 2].set_title("estDepth")
fig.colorbar(estDispAx, ax=axs[0, 1])
fig.colorbar(estDepthAx, ax=axs[0, 2])

axs[1, 0].imshow(imgL, 'gray')
trueDispAx = axs[1, 1].imshow(trueDisp, aspect='equal', cmap='viridis')
axs[1, 1].set_title("trueDisp")
trueDepthAx = axs[1, 2].imshow(trueDepth, aspect='equal', cmap='plasma')
axs[1, 2].set_title("trueDepth")
fig.colorbar(trueDispAx, ax=axs[1, 1])
fig.colorbar(trueDepthAx, ax=axs[1, 2])

plt.show()

##############################################

# f = open("classroom_depth.txt", 'w')
# for row in depth:
#     for el in row:
#         f.write(str(el) + " ")
#     f.write("\n")
# f.close()