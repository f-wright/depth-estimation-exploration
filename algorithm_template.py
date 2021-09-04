### template for range perception classes ###

import os


class RangeEstimator():
    """
        when you create an actual algorithm class, like a stereo range 
        estimator, you can inherit from this class to make things easier, e.g. 
        
        class Stereo(RangeEstimator):
            blah blah blah
    """
    def __init__(self):
        """ Calibrate the algorithm. 
        (Replace the below variables)
        """

        self.isStereo = False
        self.Algorithm = Null

    def getImage(self, basename):
        """ Returns images to be used by range estimator. 
        
        (Customize it based on whether it's monocular or stereo but follow this example format: 
            imgL = cv.imread('images/' + basename + '/im0.png', 0)
        )
        """
        pass

    def computeDisp(self, basename):
        """ Return calculated disparity in an nparray"""
        pass
        # write this method yourself

        # if your algo is monocular, just use the left img in a stereo pair.
        # if its stereo, then also add the right img.

    def computeDist(self, basename):
        """ Return calculated distance in an nparray
        """
        pass

    def getTrueDist(self, basename):
        """ Return true distance in an nparray
        """
        pass

    def computeError(self, basename):
        """ Return the MSE for the image
        """
        pass

    def computeDispMany(self, imageFolder):
        """ Return calculated disparity for all images in a folder
        """
        pass

    def computeDistMany(self, imageFolder):
        """ Return calculated distance for all images in a folder
        """
        pass

    def computeErrorMany(self, imageFolder):
        """ Return MSE for all images in a folder
        """
        pass
