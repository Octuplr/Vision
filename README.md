# Vision
UofSC|CSCE585:
A Machine Learning project to clock employees in and out using their face.

[View the paper](https://projects.cnewb.co/CSCE585/VisionPaper.pdf)


## Data
Training data is stored in `./data/AssociatePhotos/`\
Each "employee" has their own folder containing the training photos of them.\
Employee folders using this scheme: `<AssociateID>.<AssociateName>` _where `<AssociateID>` is their employee ID and `<AssociateName>` is the displayed name of that associate_.


[See Associate Photos layout.txt](data/Associate%20Photos%20layout.txt) for a diagram.


Within the encodings the associate is stored by their ID.  The VisonNet model stores associate data in sequence from 0 to _n_ ( _n_ being the number of associates). _this is done due to the model requiring n number of outputs to function_\
A key pair object is used to match associate IDs to the output of the visionNet model (using the model's `implementationData` as that object)\

At the moment the associate IDs are not mapped back to the associate names (meaning the names have no meaning past sorting your folders).


Testing data is stored in `./data/TestingPhotos/` and follows the same rules as above.\
When testing the vision model using the visionCLI, the testing data is pulled from the testing encodings file `./output/testing-encodings.dat`.\
You must rebuild this file using the `generateEncodings.py` script. This will build an encodings file for the visionNet model (->`./output/encodings.dat`) _and_ generate a similar encodings file using the testing photos (->`./output/testing-encodings.dat`)


***


## Basic usage (command line)
`python visionCLI.py` to use the command line interface tool

`python visionCLI.py -(train | t)` rebuilds the model (both embeddings and visionNet) _(visionNet training data NOT logged)_

`python visionCLI.py -(rebuild | r) (visionnet | v) [-nolog | -nl]` rebuilds the visionNet model _(if nolog then training data is not logged)_

`python visionCLI.py -(rebuild | r) (embeddings | e)` rebuilds the embeddings file

`python visionCLI.py -(punch | p) [image_path] -[noshow | ns]` to "check"/punch an associate in (if noshow is present, the image (with bounding boxes) will not be shown)

Command line interface tool: (no arguments)\
`1) Clock associate in`: Similar to the `-punch` arg, main function of the program.\
`2) Rebuild databases`: Rebuilds both the encodings and VisionNet model\
`3) Rebuild face embeddings`: Rebuilds the encodings file (update photo data for the AI)\
`4) Rebuild VisionNet model`: Rebuilds the VisionNet model (the AI)\
`5) Quit`: Quits the application

`6) Test punches`: A loop constantly calling the `punch()` function (to easily test punch-ins over-and-over)\
`7) Test accuracy`: Supplies the faces in the testing encodings file to the VisionNet model to test the number of correct prodictions (and average confidence (accuracy) level).


***

## Basic usage (API)
```python
from visionAPI import Vision
#actual
vision = Vision(dataPath = "/data/", modelName = "model.ognn", datasetDirectory = "AssociatePhotos", encodingFile = "encodings.dat", attendanceFile = "punches.csv", encoderSettings = {}, logToFile = True, logToConsole = False)

# recommend as this follows our folder structure
vision = Vision(dataPath = "output/", datasetDirectory = "../data/AssociatePhotos/", logToConsole = True)
```
`dataPath`: Path to the folder containing the embeddings file and visionNet model (in addition to the model log data)

`modelName`: File name of the model you'll be loading and using

`datasetDirectory`: The folder containing the associate training photos

`encodingFile`: File name of the embeddings you'll be loading and using (if rebuilding visionNet)

`attendanceFile`: File name containing a log (CSV) of all punches (classifications)

`encoderSettings`: Dictionary passed to the faceEncoding class (leave blank to use faceEncoding's defaults)

`logToFile`: \[not implemented\], any 'log' data is written to a file

`logToConsole`: Any 'log' data is printed to the console


/=/=/=/

#### Classifying an employee:
```python
vision.classifyAssociate(image, isFilePath = False, tolerance = 70)
```
Returns (a tuple) an array of associates in the photo and an array with the confidence levels

`image`: Either a string pointing the image OR JPG data (bytes- I.E. you opened the file and are passing that data)

`isFilePath`: Whether or not _`image`_ is a file path or data

`tolerance`: Confidence must be higher than this % to be accepted


/=/=/=/

#### Punch (clock) in/out an associate:
```python
vision.addAssociatePunch(image, isFilePath = False, drawOnImage = False, tolerance = 70)
```
Given an image, returns the employee's ID and the confidence of that labeling (and if drawOnImage, returns the image with the boxes drawn)

Returns (a tuple) array of associates in the photo, an array of their confidence levels, and the img object (if drawOnImage, so you can display the image we drew on)

`image`: JPG raw data (or if isFilePath, the (exact, from root) file path)

`ifFilePath`: Signifies if the image parameters is a string pointing to an image or if it is image data

`drawOnImage`: Whether or not we draw the bounding boxes and employee IDs on the image

`tolerance`: Confidence must be higher than this % to be accepted


/=/=/=/

#### Get neural network model information
```python
modelInfomation = vision.modelInfo()
```
Returns the model's .modelInfo() `self.neuralNetwork.modelInfo()`


/=/=/=/

#### Break the embeddings into a feature_set and one_hot_label to train the network
```python
feature_set, one_hot_labels, idMappings = vision.loadEncodings(path)
```
Builds the feature set and one_hot_labels set for taining the neural network (Prepares the embeddings so we can train visionNet on them)

`path`: The path to the embeddings file the model will be trained on


/=/=/=/

#### Build a visionNet model:
```python
trainingInfomation = vision.rebuildVisionNet(epochs = 50000, log = False, logInterval = 100, logAccuracy = True)
```
Builds the Vision net neural network. _Runs `vision.loadEncodings()` to build the feature_set and one_hot_labels in order to train._

`epochs`: Number of times the model (visionNet) gets trained

`log`: Whether or not training data is logged (do this to generate graphs and a CSV)

`logInterval`: Log every n number of epochs

`logAccuracy`: Include the average training accuracy with the log data (the average confidence and % correct)

Returns: an array with error cost (index 0) and average training accuracy (confidence) (index 1)


/=/=/=/

#### Encode the images into embeddings
```python
vision.rebuildEmbeddings()
```
Rebuilds the embeddings file (using the dataSet specified and saves to the encodingFile)


/=/=/=/

#### Rebuilds both the embeddings database and visionNet model
```python
vision.rebuild(rebuildEmbeddings = True, rebuildVisionNet = True, epochs = 50000, log = False, logInterval = 100, logAccuracy = True)
```
Runs `vision.rebuildEmbeddings()` (if `rebuildEmbeddings` is true) then runs `vision.rebuildVisionNet(epochs, log, logInterval, logAccuracy)` (if `rebuildVisionNet` is true)

_Rebuilds the embeddings file (if parameters is true) and the visionNet (if parameters is true) (and passes the visionNet parameters)_



***

## Requirements
Install Python 3+ (I have version 3.8.5)


You will need **dlib**. _[How to Install dlib](https://www.pyimagesearch.com/2017/03/27/how-to-install-dlib/)_


Requirements
_Download [CMake](https://cmake.org/download/)_
_If you running Windows, you'll have to install Visual Studio C++ packages_
`pip install cmake`
`pip install dlib` _Face detection, embeddings_

`pip install opencv-python` _May not actually need?_

`pip install numpy` _Numpy..._

`pip install pillow` _Image processing_

`pip install tqdm` _Progress bar_

`pip install matplotlib` _Plotting data_

`pip install pandas` _Logging data_


***

## Files:
`visionCLI.py`: Only needed to interact with visionAPI

*requires `visionAPI.py`*


`visionAPI.py`: Vision API (not needed as long as you implement the same functionality)

*requires `faceEncoding.py, neuralNetwork.py`*


`faceEncoding.py`: Phase 1, wrapper class that takes an image (or images), uses dlib to extract the face detail embeddings and save those to a file.

*requires data/dlib_models/\**


`neuralNetwork.py`: Phase 2, a feed-forward neural network class we use for visionNet.


`generateEncodings.py`: Generate the face embeddings from the associate images and save those to a file (deprecated, use visionAPI.py instead)

`generateVisionNet.py`: Generates a visionNet model from the embeddings file (used as a quicker and more 'debugily' way of generating a model, use visionAPI.py instead)



`extras/vision.py`: Deprecated, version one of the project, (only uses embeddings and k-nearest neighbor)

`extras/face_recognition.py`: Deprecated, version one dependency (slightly modified from official package)

`extras/cropPhoto.py`: Don't know why this is here, python script to crop a photo and extract the faces out of the photo.
