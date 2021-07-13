### overview:

clone the repository and install the dependencies using

```pip install -r requirements.txt

```

Populate the images/ directory with any image data you want to use.
Each image subfolder should contain at least the following data:

- calib.txt, containing at least fx, baseline, and doffs (see Middlebury dataset for example of format)
- disp0.pfm, containing ground truth
- img0.png and img1.png, the left and right stereo images respectively.
