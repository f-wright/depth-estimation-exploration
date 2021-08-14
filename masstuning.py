import numpy as np
import cv2 as cv
import os
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider, Button
from pfm_conversion import read_pfm

basename = 'Bicycle1-perfect'
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
    'BS': 15,  #BlockSize
}


def get_est_disp(basename):
    """estimated disparity using stereo"""
    imgL = cv.imread('images/' + basename + '/im0.png', 0)  # type Mat
    imgR = cv.imread('images/' + basename + '/im1.png', 0)

    ##### setup stereoBM
    sbm = cv.StereoBM_create(numDisparities=16, blockSize=15)
    #sbm.SADWindowSize = SWS
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(sbmParams['PFS'])
    sbm.setPreFilterCap(sbmParams['PFC'])
    sbm.setMinDisparity(sbmParams['MDS'])
    sbm.setNumDisparities(sbmParams['NOD'])
    sbm.setTextureThreshold(sbmParams['TTH'])
    sbm.setUniquenessRatio(sbmParams['UR'])
    sbm.setSpeckleRange(sbmParams['SR'])
    sbm.setSpeckleWindowSize(sbmParams['SPWS'])
    sbm.setBlockSize(sbmParams['BS'])

    disparity = sbm.compute(imgL, imgR) / 16
    return disparity


def get_true_disp(basename):
    """return true disparity"""
    true_disp = read_pfm('images/' + basename + '/disp0.pfm')
    return true_disp


def gather_all_img_data():
    """returns dict with all image data"""
    img_list = []

    print('loading image data.....')
    for filename in os.listdir("images/"):
        if filename != '.DS_Store':
            print(filename)
            imgL = cv.imread('images/' + filename + '/im0.png', 0)
            est_disparity = get_est_disp(filename)
            true_disparity = get_true_disp(filename)
            # store data in a list
            img_list.append([filename, imgL, est_disparity, true_disparity])

    return img_list


def draw_comparison_window():
    """compares estimated and true disparities for all images in folder."""

    img_list = gather_all_img_data()

    # Set up and draw interface
    print('Set up and draw interface')
    axcolor = 'lightgoldenrodyellow'
    numImgs = len(img_list)
    print(str(numImgs) + " images")
    fig, axs = plt.subplots(3, numImgs)
    plt.subplots_adjust(bottom=0.4)

    # iterate & draw all images, estd disparities, and true disparities

    axes = []
    for i in range(numImgs):
        basename = img_list[i][0]
        img = img_list[i][1]
        estDisp = img_list[i][2]
        trueDisp = img_list[i][3]

        print("displaying " + basename)

        ax0 = axs[0, i].imshow(img, 'gray')
        ax1 = axs[1, i].imshow(trueDisp, aspect='equal', cmap='viridis')
        ax2 = axs[2, i].imshow(estDisp, aspect='equal', cmap='viridis')
        axs[0, i].axis('off')
        axs[1, i].axis('off')
        axs[2, i].axis('off')

        axes.append([ax0, ax1, ax2])

    # draw axes
    SWSaxe = plt.axes([0.15, 0.01, 0.7, 0.025],
                      facecolor=axcolor)  #stepX stepY width height
    PFSaxe = plt.axes([0.15, 0.05, 0.7, 0.025],
                      facecolor=axcolor)  #stepX stepY width height
    PFCaxe = plt.axes([0.15, 0.09, 0.7, 0.025],
                      facecolor=axcolor)  #stepX stepY width height
    MDSaxe = plt.axes([0.15, 0.13, 0.7, 0.025],
                      facecolor=axcolor)  #stepX stepY width height
    NODaxe = plt.axes([0.15, 0.17, 0.7, 0.025],
                      facecolor=axcolor)  #stepX stepY width height
    TTHaxe = plt.axes([0.15, 0.21, 0.7, 0.025],
                      facecolor=axcolor)  #stepX stepY width height
    URaxe = plt.axes([0.15, 0.25, 0.7, 0.025],
                     facecolor=axcolor)  #stepX stepY width height
    SRaxe = plt.axes([0.15, 0.29, 0.7, 0.025],
                     facecolor=axcolor)  #stepX stepY width height
    SPWSaxe = plt.axes([0.15, 0.33, 0.7, 0.025],
                       facecolor=axcolor)  #stepX stepY width height
    BSaxe = plt.axes([0.15, 0.37, 0.7, 0.025],
                     facecolor=axcolor)  #stepX stepY width height

    sSWS = Slider(SWSaxe, 'SWS', 5.0, 255.0, valinit=sbmParams['SWS'])
    sPFS = Slider(PFSaxe, 'PFS', 5.0, 255.0, valinit=sbmParams['PFS'])
    sPFC = Slider(PFCaxe, 'PreFiltCap', 5.0, 63.0, valinit=sbmParams['PFC'])
    sMDS = Slider(MDSaxe, 'MinDISP', -100.0, 100.0, valinit=sbmParams['MDS'])
    sNOD = Slider(NODaxe, 'NumOfDisp', 16.0, 320.0, valinit=sbmParams['NOD'])
    sTTH = Slider(TTHaxe, 'TxtrThrshld', 0.0, 1000.0, valinit=sbmParams['TTH'])
    sUR = Slider(URaxe, 'UnicRatio', 1.0, 20.0, valinit=sbmParams['UR'])
    sSR = Slider(SRaxe, 'SpcklRng', 0.0, 40.0, valinit=sbmParams['SR'])
    sSPWS = Slider(SPWSaxe,
                   'SpklWinSze',
                   0.0,
                   300.0,
                   valinit=sbmParams['SPWS'])
    sBS = Slider(BSaxe, 'BlockSize', 0.0, 40.0, valinit=sbmParams['BS'])

    def update_est_disparities():
        for i in range(numImgs):
            basename = img_list[i][0]
            # print('recalculating ' + filename)
            est_disparity = get_est_disp(basename)
            # update list
            img_list[i][2] = est_disparity
            # redraw it
            ax2 = axes[i][2]
            ax2.set_data(est_disparity)

    def update(val):
        sbmParams['SWS'] = int(sSWS.val / 2) * 2 + 1  #convert to ODD
        sbmParams['PFS'] = int(sPFS.val / 2) * 2 + 1
        sbmParams['PFC'] = int(sPFC.val / 2) * 2 + 1
        sbmParams['MDS'] = int(sMDS.val)
        sbmParams['NOD'] = int(sNOD.val / 16) * 16
        sbmParams['TTH'] = int(sTTH.val)
        sbmParams['UR'] = int(sUR.val)
        sbmParams['SR'] = int(sSR.val)
        sbmParams['SPWS'] = int(sSPWS.val)
        sbmParams['BS'] = int(sBS.val)

        print('Rebuilding depth map')

        update_est_disparities()
        print('Redraw depth map')
        plt.draw()

    # Connect update actions to control elements
    sSWS.on_changed(update)
    sPFS.on_changed(update)
    sPFC.on_changed(update)
    sMDS.on_changed(update)
    sNOD.on_changed(update)
    sTTH.on_changed(update)
    sUR.on_changed(update)
    sSR.on_changed(update)
    sSPWS.on_changed(update)
    sBS.on_changed(update)

    plt.show()


if __name__ == "__main__":
    draw_comparison_window()
