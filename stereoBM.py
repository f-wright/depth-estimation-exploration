import os
from algorithm_template import RangeEstimator
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import re


class StereoBM(RangeEstimator):
    def __init__(self):
        """ Calibrate the algorithm. 
        """

        self.isStereo = True

        # initialize the stereoBM object
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

        self.Algorithm = sbm

    def getImage(self, basename):
        """ Returns image(s) to be used by range estimator. If the algorithm is stereo, return a tuple of left and right images. If it's monocular, return only the left image.)
        
        (Customize it based on whether it's monocular or stereo but follow this example format: 
            imgL = cv.imread('images/' + basename + '/im0.png', 0)
        )
        """
        imgL = cv.imread('images/' + basename + '/im0.png', 0)  # type Mat
        imgR = cv.imread('images/' + basename + '/im1.png', 0)

        return imgL, imgR

    def computeDisp(self, basename):
        """ Return calculated disparity in an nparray
        """
        imgL, imgR = self.getImage(basename)
        estDisp = self.Algorithm.compute(imgL, imgR) / 16
        return estDisp

    def computeDist(self, basename):
        """ Return calculated distance in an nparray
        """
        fx, baseline, doffs = self.getCalibData(basename)
        estDisp = self.computeDisp(basename)
        estDepth = np.zeros(shape=estDisp.shape).astype(float)  # initialize np
        estDepth[estDisp > 0] = (fx * baseline) / (doffs + estDisp[estDisp > 0]
                                                   )  # populate np
        return estDepth

    def getTrueDist(self, basename):
        """ Return true distance in an nparray
        """
        fx, baseline, doffs = self.getCalibData(basename)
        trueDisp = self.read_pfm('images/' + basename + '/disp0.pfm')
        trueDisp = np.asarray(trueDisp)
        trueDepth = np.zeros(shape=trueDisp.shape).astype(
            float)  # initialize np
        trueDepth[trueDisp > 0] = (fx * baseline) / (
            doffs + trueDisp[trueDisp > 0])  # populate np

    def computeError(self, basename):
        """ Return the MSE for the image
        """
        pass

    def computeMany(self, imageFolder):
        """ 
        """
        pass

    ############################################################################
    # Define other helper functions as needed for your algorithm
    ############################################################################

    def getCalibData(self, basename):
        calibPath = 'images/' + basename + '/calib.txt'
        calibData = {}
        with open(calibPath) as f:
            for line in f:
                (key, val) = line.strip().split('=')
                calibData[key] = val
        fx = float(calibData['cam0'][1:].split()[0])
        baseline = float(calibData['baseline'])
        doffs = float(calibData['doffs'])
        return fx, baseline, doffs

    def read_pfm(self, basename):
        pfm_file_path = 'images/' + basename + '/calib.txt'
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
        # png_file_path = pfm_file_path[:-4] + ".png"
        # plt.imsave(os.path.join(png_file_path), img)

        # return disparity, [(height, width, channels), scale]
        return img
