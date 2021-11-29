# Vision
The Face detector and recognizer MLS

## Basic usage (command line)
`python visionCLI.py` to use the command line interface tool

`python visionCLI.py -train` to train the model

`python visionCLI.py -punch [image_path]` to "check"/punch an associate in

## Basic usage (API)
```python
from visionAPI import Vision
vision = Vision(dataPath = "output/", datasetDirectory = "../data/AssociatePhotos/", logToConsole = True)
```

#### Classifing an employee:
```python
vision.classifyAssociate(image, isFilePath = False, tolerance = 70)
```
Returns (a tuple) an array of associates in the photo and an array with the confidence levels
image: Either a string pointing the image OR JPG data (I.E. you opened the file already and are passing the data)
isFilePath: Whether or not _image_ is a file path or data
tolerance: Confidence must be higher than this % to be accepeted


#### Punch (clock) in/out an assocaite:
```python
vision.addAssociatePunch(image, isFilePath = False, drawOnImage = False, tolerance = 70)
```
Given an image, returns the employee's ID and the confidence of that labeling (and if drawOnImage, returns the image with the boxes drawn)
image: JPG raw data (or if isFilePath, the file path)
ifFilePath: Signifies if the image object is a string pointing to an image or if image is image data
drawOnImage: Whether or not we draw the bounding boxes and employee IDs on the image
tolerance: Confidence must be higher than this % to be accepted

#### Get neural network model infromation
```python
modelInfomation = vision.modelInfo()
```
Returns the model's .modelInfo() `self.neuralNetwork.modelInfo()`

#### Break the embeddings into a feature_set and one_hot_label to train the network
```python
feature_set, one_hot_labels, idMappings = vision.loadEncodings(path)
```
Builds the feature set and one_hot_labels set for taining the neural network
path: THe path to the embeddings file the model will be trained on

#### Build a visionNet model:
```python
trainingInfomation = vision.rebuildVisionNet(epochs = 50000, log = False, logInterval = 100, logAccuracy = True)
```
Builds the Vision net neural network. Runs `vision.loadEncodings()` to build the feature_set and one_hot_labels inorder to train.
epochs: Number of times the model gets trained
log: Whether or not training data is logged
logInterval: Log every n number of epochs
logAccuracy: Include the average training accuracy with the log data
Returns: an array with error cost (index 0) and average training accuracy (index 1)

#### Encode the images into embeddings
```python
vision.rebuildEmbeddings()
```
Rebuilds the embeddings file (using the dataSet specified and saves to the encodingFile)

#### Rebuilds both the embeddings database and visionNet model
```python
vision.rebuild(rebuildEmbeddings = True, rebuildVisionNet = True, epochs = 50000, log = False, logInterval = 100, logAccuracy = True)
```
Runs `vision.rebuildEmbeddings()` (if `rebuildEmbeddings` is true) then runs `vision.rebuildVisionNet(epochs, log, logInterval, logAccuracy)` (if `rebuildVisionNet` is true)





## Requirements
Install Python 3+ (I have version 3.8.5)


You will need **dlib**. _[How to Install dlib](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/)_


Requirements
_Download [CMake](https://cmake.org/download/)_
_If you running Windows, you'll have to install Visual Studio C++ packages_
`pip install cmake`
`pip install dlib` _Face detection, embeddings_

`pip install opencv-python`

`pip install numpy` _Numpy..._

`pip install pillow` _Image processing_

`pip install tqdm` _Progress bar_

`pip install matplotlib` _Plotting data_

`pip install pandas` _Logging data_
