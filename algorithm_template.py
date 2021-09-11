### template for range perception classes ###

import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import re


class RangeEstimator():
    """
        when you create an actual algorithm class, like a stereo range 
        estimator, you can inherit from this class to make things easier, e.g. 
        
        class Stereo(RangeEstimator):
            blah blah blah
    """
    def __init__(self):
        """ Initialize and calibrate the algorithm.
        (Replace the below variables)
        """

        self.isStereo = False
        self.Algorithm = Null

    def getImage(self, basename):
        """ Returns images to be used by range estimator. 
        
        (Customize based on whether it's monocular or stereo but follow the example format. Set imgR = Null if algorithm is monocular.)
        """
        imgL = cv.imread('images/' + basename + '/im0.png', 0)  # type Mat
        imgR = cv.imread('images/' + basename + '/im1.png', 0)
        # imgR = Null # (if monocular)

        return imgL, imgR

    def estimateRange(self, basename):
        """ Return estimated distance in an nparray"""
        # rewrite this method yourself

        estDisp = self.computeDisp(basename)
        estRange = np.zeros(shape=estDisp.shape).astype(float)  # initialize np

        # compute estRange yourself

        return estRange

    def getTrueRange(self, basename):
        """ Return true distance in an nparray"""
        trueDisp = self.read_pfm('images/' + basename + '/disp0.pfm')
        trueDisp = np.asarray(trueDisp)
        trueRange = np.zeros(shape=trueDisp.shape).astype(
            float)  # initialize np

        # compute estRange yourself

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
