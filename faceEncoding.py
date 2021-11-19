import PIL.Image
import dlib
import numpy as np
from pkg_resources import resource_filename


'''
Vision -
Phase 1 

Image -> im = PIL.Image.open(file)
	if mode:
		im = im.convert(mode)
	return np.array(im)

	||||
	vvvv

Find face(s) (`face_location` bounding boxes) -> dlib.get_frontal_face_detector()(image, number_of_times_to_upsample)

	||||
	vvvv

Plot face(s) landmarks -> dlib.shape_predictor(68_point_model)(image, face_locations)

	||||
	vvvv

Compute face encodings -> dlib.face_recognition_model_v1(face_recognition_model)(image, face_landmarks, num_jitters)

	||||
	vvvv

Encodings


dlib models provided by: https://github.com/davisking/dlib-models

dlib_face_recognition_resnet_model_v1.dat: Generating encodings

shape_predictor_68_face_landmarks_GTX.dat: Finding face landmarks
shape_predictor_5_face_landmarks.dat: Finding face landmarks (smaller model)

mmod_human_face_detector.dat: Detect and find face locations (rather than using hog)

'''

class faceEncoder():
	def __init__(self, shapeModelSize = "large", faceDetectorModel = "hog", numJitters = 1, numUpsamples = 1, numpyLibrary = np):
		np = numpyLibrary
		if shapeModelSize == "large":
			self.shapeModelSize = "large"
			self.shapePredictor = dlib.shape_predictor("data/dlib_models/shape_predictor_68_face_landmarks_GTX.dat")
		else:
			self.shapeModelSize = "small"
			self.shapePredictor = dlib.shape_predictor("data/dlib_models/shape_predictor_5_face_landmarks.dat")

		if faceDetectorModel == "hog":
			self.faceDetectorModel = "hog"
			self.faceDetector = dlib.get_frontal_face_detector()
		else:
			self.faceDetectorModel = "cnn"
			self.faceDetector = dlib.cnn_face_detection_model_v1("data/dlib_models/mmod_human_face_detector.dat")


		self.encoder = dlib.face_recognition_model_v1("data/dlib_models/dlib_face_recognition_resnet_model_v1.dat")

		self.numberOfJitters = numJitters
		if (numJitters<1):
			self.numberOfJitters = 1
		self.numberOfUpsamples = numUpsamples
		if (numUpsamples<1):
			self.numberOfUpsamples = 1

	def rectangleToTuple(self, rect):
		return rect.top(), rect.right(), rect.bottom(), rect.left()

	def tupleToRectangle(self, rect):
		return dlib.rectangle(rect[3], rect[0], rect[1], rect[2])

	def trimRectangleToBounds(self, rect, imageShape):
		return max(rect[0], 0), min(rect[1], imageShape[1]), min(rect[2], imageShape[0]), max(rect[3], 0)


	def encodingDistance(self, encodings, unknown):
		if len(encodings) == 0:
			return np.empty((0))

		return np.linalg.norm(encodings - unknown, axis=1)

	def loadImageFile(self, file, mode = "RGB"):
		im = (PIL.Image.open(file)).convert(mode)
		# return np.array(im)
		self.image = np.array(im)
		return self.image

	def faceLocationsRaw(self, image = None):
		if image is None:
			image = self.image
		try:
			return self.faceDetector(image, self.numberOfUpsamples)
		except MemoryError:
			raise MemoryError("Not enough memory, too many upsamples?")

	def faceLocations(self, image = None):
		if image is None:
			image = self.image
		if (self.faceDetectorModel == "hog"):
			return [self.trimRectangleToBounds(self.rectangleToTuple(faceLocation), image.shape) for faceLocation in self.faceLocationsRaw(image)]
		else:
			return [self.trimRectangleToBounds(self.rectangleToTuple(faceLocation.rect), image.shape) for faceLocation in self.faceLocationsRaw(image)]

	def faceLandmarks(self, image = None, faceLocations = None):
		if image is None:
			image = self.image
		if faceLocations == None:
			faceLocations = self.faceLocationsRaw(image)
		else:
			faceLocations = [self.tupleToRectangle(faceLocation) for faceLocation in faceLocations]

		return [self.shapePredictor(image, faceLocation) for faceLocation in faceLocations]

	def encode(self, image = None, faceLocations = None):
		if image is None:
			image = self.image
		landmarks = self.faceLandmarks(image, faceLocations)
		return [np.array(self.encoder.compute_face_descriptor(image, landmarkSet, self.numberOfJitters)) for landmarkSet in landmarks]