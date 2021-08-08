### overview:

- clone the repository
- create a virtual environment (optional but recommended):

```
virtualenv venv
source venv/bin/activate
```

install the dependencies:

```
pip install -r requirements.txt
```

- Populate the images/ directory with any image data you want to use.
  Each image subfolder should contain at least the following data:

  - calib.txt, containing at least fx, baseline, and doffs (see Middlebury dataset for example of format)
  - disp0.pfm, containing ground truth
  - img0.png and img1.png, the left and right stereo images respectively.

- run the code, e.g.

```
python3 stereo2.py
```
