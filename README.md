# Biometric Identification Using Principal Lines on a Palm

This project explores biometric identification using principal lines on human palms. The model extracts rotation-invariant region of the palm from the source image, filters the region to highlight ridges and troughs and extracts a set of 9 floats according to which nearest neighbor classification is performed.

## Running locally

1. Make sure you have Python 3.10 installed
2. Install requirements listed in `requirements.txt`
3. Download [dataset](http://biometrics.idealtest.org/dbDetailForUser.do?id=6#/datasetDetail/6) and place its contents into `images/` folder
4. Run `evaluate.py` to evaluate model on dataset
