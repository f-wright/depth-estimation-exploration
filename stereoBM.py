import os
from algorithm_template import RangeEstimator
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from matplotlib import colors
import re


class StereoBM(RangeEstimator):
    def __init__(self):
        """ Initialize and calibrate the algorithm."""

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

    def estimateRange(self, basename):
        """ Return estimated distance in an nparray"""
        estDisp = self.estimateDisp(basename)
        estRange = self.dispToRange(basename, estDisp)

        return estRange

    def getTrueRange(self, basename):
        """ Return true distance in an nparray"""
        trueDisp = self.read_pfm(basename)
        trueDisp = np.asarray(trueDisp)
        trueRange = self.dispToRange(basename, trueDisp)

        return trueRange

    def computeDiff(self, basename):
        """ Return the MSE for the image"""
        estRange = self.estimateRange(basename)
        trueRange = self.getTrueRange(basename)
        diff = np.subtract(estRange, trueRange)

        return diff

    def computeMSE(self, basename):
        diff = self.computeDiff(basename)
        sqdiff = np.square(diff)
        mse = np.ndarray.mean(sqdiff)

        return mse

    def estimateRangeMany(self, imageFolder):
        """ Return estimated distance for all images in a folder
        """
        pass

    def computeErrorMany(self, imageFolder):
        """ Return MSE for all images in a folder
        """
        pass

    def plot(self, basename):
        """ visualize true range, estimated range, and error"""
        imgL, imgR = self.getImage(basename)
        estRange = self.estimateRange(basename)
        trueRange = self.getTrueRange(basename)
        print(estRange.dtype)
        print(trueRange.dtype)
        diff = np.subtract(estRange, trueRange)
        mse = self.computeMSE(basename)

        images = []

        fig, axs = plt.subplots(1, 4)
        fig.suptitle(
            basename + "; MSE = " + str(mse) +
            "; note: pink error is underestimated, green is overestimated")

        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')
        axs[3].axis('off')

        axs[0].imshow(imgL, 'gray')

        axs[1].set_title("estRange")
        estRangeAx = axs[1]
        images.append(estRangeAx.imshow(estRange, cmap='plasma'))

        axs[2].set_title("trueRange")
        trueRangeAx = axs[2]
        images.append(trueRangeAx.imshow(trueRange, cmap='plasma'))

        axs[3].set_title("error")
        errorAx = axs[3].imshow(diff, 'PiYG')

        cax1 = plt.axes([0.3, 0.7, 0.3, 0.01])
        cax2 = plt.axes([0.9, 0.4, 0.01, 0.3])
        vmin = min(image.get_array().min() for image in images)
        vmax = max(image.get_array().max() for image in images)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        for im in images:
            im.set_norm(norm)

        fig.colorbar(images[0], cax=cax1, orientation='horizontal')
        fig.colorbar(errorAx, cax=cax2)

        plt.show()

    def plotMany(self, imageFolder):
        """ visualize true range, estimated range, and error for all images in 
            a folder. 
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

    def estimateDisp(self, basename):
        """ Return estimated disparity in an nparray"""
        imgL, imgR = self.getImage(basename)
        estDisp = self.Algorithm.compute(imgL, imgR) / 16
        return estDisp

    def dispToRange(self, basename, disp):
        fx, baseline, doffs = self.getCalibData(basename)
        range = np.zeros(shape=disp.shape).astype(float)  # initialize np
        range[disp > 0] = (fx * baseline) / (doffs + disp[disp > 0]
                                             )  # populate np
        return range

    def read_pfm(self, basename):
        pfm_file_path = 'images/' + basename + '/disp0.pfm'
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

        img = np.reshape(disparity, newshape=(height, width))
        img = np.flipud(img).astype('uint8')
        #
        # plt.show(img, "disparity")
        # png_file_path = pfm_file_path[:-4] + ".png"
        # plt.imsave(os.path.join(png_file_path), img)

        # return disparity, [(height, width, channels), scale]
        return img


# for testing purposes
if __name__ == "__main__":
    stereoTest = StereoBM()
    basename = "Backpack-imperfect"
    stereoTest.plot(basename)