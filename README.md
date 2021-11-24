# Vision
The Face detector and recognizer MLS

### Basic usage (command line)
`python visionCLI.py` to use the command line interface tool

`python visionCLI.py -train` to train the model

`python visionCLI.py -punch [image_path]` to "check"/punch an associate in

### Basic usage (API)
```python
from visionAPI import Vision
vision = Vision(dataPath = "output/", datasetDirectory = "../data/AssociatePhotos/", logToConsole = True)
```

(Vision API doc goes here. . . for now just read the visionAPI.py file)


### Requirements
Install Python 3+ (I have version 3.8.5)


You will need **dlib**. _[How to Install dlib](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/)_


Install opencv
`pip install opencv-python`
