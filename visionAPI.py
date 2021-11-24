from neuralNetwork import genericNeuralNetwork
from faceEncoding import faceEncoder

import numpy as np

import os # filesystem, check/make dir
import PIL # Image
from PIL import ImageDraw
from io import BytesIO # JPG raw to image
import datetime
import time

import pickle # embedding management

from tqdm import tqdm # Progress bar
from copy import deepcopy # evaluate model during training by making exact copy
import matplotlib.pyplot as plt # for plotting
import pandas as pd # Logging


def ga(model, data, output):
	acc = 0
	acc2 = 0
	accuracyFigures = []
	for xx,yy in zip(data, output):
		s = model.predictRaw(np.array([xx]))
		predictionLabel = s.argmax()
		accuracy =  round(float(s[0][predictionLabel])*100, 4)
		accuracyFigures.append([np.argmax(yy), predictionLabel, accuracy])
		if predictionLabel == np.argmax(yy):
			acc+=1
			acc2 = acc2 + accuracy

	return [(acc/len(data)*100), (acc2/len(data)), accuracyFigures]

class Vision():
	def __init__(self, dataPath = "/data/", modelName = "model.ognn", datasetDirectory = "AssociatePhotos", encodingFile = "encodings.dat", attendanceFile = "punches.csv", encoderSettings = {}, logToFile = True, logToConsole = False, useCupy = False):
		#// to do: add directory checking
		self.dataPath = dataPath
		self.modelName = modelName
		self.datasetDirectory = datasetDirectory
		self.encodingFile = encodingFile
		self.attendanceFile = attendanceFile
		self.logToFile = logToFile
		self.logToConsole = logToConsole

		self.neuralNetwork = genericNeuralNetwork() # We don't have to pass any parameters since we load a built model. (We can reinitialize the class if we want to build a new model)
		# load that mentioned model
		self.neuralNetwork.loadFile(self.dataPath + self.modelName)

		self.encoderSettings = encoderSettings
		if not "shapeModelSize" in encoderSettings:
			self.encoderSettings["shapeModelSize"] = "large"
		if not "faceDetectorModel" in encoderSettings:
			self.encoderSettings["faceDetectorModel"] = "cnn"
		if not "numJitters" in encoderSettings:
			self.encoderSettings["numJitters"] = 10
		if not "numUpsamples" in encoderSettings:
			self.encoderSettings["numUpsamples"] = 0
		self.encoder = faceEncoder(shapeModelSize = self.encoderSettings["shapeModelSize"], faceDetectorModel = self.encoderSettings["faceDetectorModel"], numJitters = self.encoderSettings["numJitters"], numUpsamples = self.encoderSettings["numUpsamples"])

		with open(self.dataPath + self.encodingFile, 'rb') as f:
			self.embeddings = pickle.load(f)

	def log(self, msg):
		if self.logToConsole:
			print(msg)

		# append to log file?


	def refreshEmbeddings(self):
		with open(self.dataPath + self.encodingFile, 'rb') as f:
			self.embeddings = pickle.load(f)


	def classifyAssociate(self, image, isFilePath = False, tolerance = 70):
		"""
		Given an image, returns the employee's ID and the confidence of that labeling
		:param image: JPG raw data (or if isFilePath, the file path)
		:param drawOnImage: Whether or not we draw the bounding boxes and employee IDs on the image
		:param tolerance: Confidence must be higher than this % to be accepted

		"""
		img
		if isFilePath:
			img = PIL.Image.open(file)
		else:
			# Imma assume JPG data
			img = PIL.Image.open(BytesIO(image))
		
		img = img.convert("RGB")
		image = np.array(img)

		unknownEmbeddings = self.encoder.encode(image)
		associates = []
		confidences = []
		for unknownFaceEmbedding in unknownEmbeddings:
			rawPredict = self.neuralNetwork.predictRaw([unknownFaceEmbedding]) # Returns the raw label predictions
			index = rawPredict.argmax() # The label with the highest confidence
			label = int(self.neuralNetwork.implementationData[index]) # The employee's actual ID (output classification ->>mapped>>- employee ID)
			confidence = round((rawPredict[0][index])*100, 2) # Confidence (accuracy)
			if (confidence >= tolerance):
				# A match
				associates.append(label)
				confidences.append(confidence)
		return associates, confidences

	def addAssociatePunch(self, image, isFilePath = False, drawOnImage = False, tolerance = 70):
		ts = time.time() # This is the current time. Do not penalize the associate for the system processing time.
		img = None
		if isFilePath:
			if os.path.isfile(image):
				img = PIL.Image.open(image)
			else:
				return
		else:
			# Imma assume JPG data
			from io import BytesIO
			img = PIL.Image.open(BytesIO(image))
		img = img.convert("RGB")
		image = np.array(img)

		faceLocations = self.encoder.faceLocations(image)
		unknownEmbeddings = self.encoder.encode(image, faceLocations)

		if drawOnImage:
			draw = ImageDraw.Draw(img)

		# Load attendance file

		if os.path.isfile(self.dataPath + self.attendanceFile):
			attendanceFile = open(self.dataPath + self.attendanceFile, "w")
			attendanceFile.write("ID, date, time, confidence\n")
		else:			
			attendanceFile = open(self.dataPath + self.attendanceFile, "a")


		date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
		timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')

		cords = {}
		associates = []
		confidences = []
		for (top, right, bottom, left), unknownFaceEmbedding in zip(faceLocations, unknownEmbeddings):
			kNNDistance = self.encoder.encodingDistance(np.array(list(self.embeddings.values())), unknownFaceEmbedding)

			rawPredict = self.neuralNetwork.predictRaw([unknownFaceEmbedding]) # Returns the raw label predictions
			index = rawPredict.argmax() # The label with the highest confidence
			label = int(self.neuralNetwork.implementationData[index]) # The employee's actual ID (output classification ->>mapped>>- employee ID)
			confidence = round((rawPredict[0][index])*100, 2) # Confidence (accuracy)

			if (confidence >= tolerance):
				# A match
				self.log("Associate punch added for ID " + str(label) + " (confidence: " + str(confidence) + "%).")
				associates.append(label)
				confidences.append(confidence)
				
				attendanceFile.write(str(label) + ", " + str(date) + ", " + str(timeStamp) + ", " + str(confidence) + "\n")

				if drawOnImage:
					draw.rectangle( ((left, top), (right, bottom)), outline=(50, 255, 50)) # Green box (identified)
					draw.rectangle( ((left, bottom - 20), (right, bottom)), fill=(50, 255, 50), outline=(50, 255, 50))
					draw.text((left + 3, bottom - 13), "ID: " + str(label) + " (" + str(confidence) + "%)", fille=(200, 200, 200, 255))
			else:
				self.log("Low/Unknown face detected (possible ID: " + str(label) + ", confidence: " + str(confidence) + "%).")
				if drawOnImage:
					draw.rectangle( ((left, top), (right, bottom)), outline=(255, 50, 50)) # Red box (unidentified)
					draw.rectangle( ((left, bottom - 20), (right, bottom)), fill=(255, 50, 50), outline=(255, 50, 50))
					draw.text((left + 3, bottom - 13), "~!ID?: " + str(label) + " (" + str(confidence) + "%)", fille=(200, 200, 200, 255))

		#
		if drawOnImage:
			timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H-%M-%S')
			date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
			img.save(self.dataPath + "Punches/Punch_"+date+"_"+timeStamp+".png", format="PNG")

		attendanceFile.close()
		del draw

		if drawOnImage:
			return associates, confidences, img
		else:
			return associates, confidences

	


	# Model specific
	def modelInfo(self):
		return self.neuralNetwork.modelInfo()


	# Build/Rebuild database:
	def loadEncodings(self, path):
		if path is not None:
			print(path)
			with open(path, 'rb') as f:
				self.embeddings = pickle.load(f)

		face_encodings = {}

		for key in self.embeddings:
			encodingId = key.split(":")[0]
			if encodingId in face_encodings:
				face_encodings[encodingId].append(self.embeddings[key]) # Append
			else:
				face_encodings[encodingId] = [self.embeddings[key]]

		# Build feature set

		idMappings = {}
		idMappingsReversed = {}
		freeIndex = 0
		for key in face_encodings:
			if key in idMappings:
				pass
			else:
				idMappings[freeIndex] = key
				idMappingsReversed[key] = freeIndex
				freeIndex = freeIndex+1

		feature_set_builder = None
		labels = None
		numberOfEncodings = 0
		self.numberOfClassifications = 0
		for key in face_encodings:
			numberOfEncodings = len(face_encodings[key]) + numberOfEncodings
			self.numberOfClassifications = self.numberOfClassifications + 1
			if feature_set_builder is None:
				feature_set_builder = np.array(face_encodings[key])
				labels = np.array([idMappingsReversed[key]]*len(face_encodings[key]))
			else:
				feature_set_builder = np.concatenate((feature_set_builder, np.array(face_encodings[key])))
				labels = np.concatenate((labels, np.array([idMappingsReversed[key]]*len(face_encodings[key]))))

		feature_set = np.vstack(feature_set_builder)

		# Build output label
		one_hot_labels = np.zeros((numberOfEncodings, self.numberOfClassifications))

		for i in range(numberOfEncodings):
			one_hot_labels[i, labels[i]] = 1

		return feature_set, one_hot_labels, idMappings

	def rebuildVisionNet(self, epochs = 50000, log = False, logInterval = 100, logAccuracy = True):
		"""
		Loads an image file (.jpg, .png, etc) into a numpy array

		:param epochs: image file name or file object to load
		:param log: Whether or not training data is logged
		:param logInterval: Log every n number of epochs
		:param logAccuracy: Include the average training accuracy with the log data
		:return: array with error cost (index 0) and average training accuracy (index 1)
		"""
		# ehh uhhh, a more refined generateVisionNet.py
		feature_set, one_hot_labels, idMappings = self.loadEncodings(self.dataPath + self.encodingFile)
		self.neuralNetwork = genericNeuralNetwork(np.array(feature_set), one_hot_labels, neurons = int(self.numberOfClassifications*8), implementationData = idMappings)

		error_cost = []
		tAccuracy = []
		col_names = ['Loss Function','Training correctness','Average training accuracy']
		logCSV = pd.DataFrame(columns = col_names)
		
		if self.logToConsole:
			self.neuralNetwork.printInfo()
			print("\n")
		for x in tqdm(range(epochs), desc="Training model", colour="green", ascii=True, leave=False):
		# for x in range(epochs):
			self.neuralNetwork.feedforward()
			self.neuralNetwork.backprop()
			if log:
				if x % logInterval == 0:
					loss = self.neuralNetwork.getLoss()
					error_cost.append(loss)
					acc = None
					if logAccuracy == True:
						acc = ga(deepcopy(self.neuralNetwork), feature_set, one_hot_labels)
						tAccuracy.append(acc[1])
					logCSV.loc[len(logCSV)] = [loss, acc[0], acc[1]]

		acc = ga(self.neuralNetwork, feature_set, one_hot_labels)
		self.log("Training correctness:    " + str(acc[0]))
		self.log("Training avg confidence: " + str(acc[1]))
		
		fileName = str(self.neuralNetwork.attributes) + "-input_" + str(self.neuralNetwork.numberOfHiddenLayers+2) + "-layer_" + str(self.neuralNetwork.neurons) + "-neurons_" + str(self.neuralNetwork.output_labels) + "-classifications_model-ID_" + self.neuralNetwork.modelID

		# Save model
		self.neuralNetwork.save(self.dataPath + "model_data/" + fileName + ".ognn") # Saves without activations or input data, but with the output data?
		self.neuralNetwork.save(self.dataPath + self.modelName) # Saves without activations or input data, but with the output data?
		

		# Log to CSV
		if log:
			logCSV.to_csv(self.dataPath + "model_data/" + fileName + ".csv",index=False)

			# Plot
			fig, ax = plt.subplots()
			fig.suptitle("Model results")
			ax.set_title(fileName)
			ax.set_xlabel("Run (x" + str(logInterval) + ")")
			ax.plot(npA.arange(1,len(error_cost)+1), error_cost, color="blue")
			ax.set_ylabel("Loss", color="blue")

			if logAccuracy:
				ax2 = plt.twinx()
				ax2.plot(npA.arange(1,len(tAccuracy)+1), tAccuracy, color="orange")
				# ax2.plot(np.arange(1,len(vAccuracy)+1), vAccuracy, color="brown")
				ax2.set_ylabel("Average accuracy %", color="orange")
			
			plt.savefig(self.dataPath + "model_data/" + fileName + ".svg")
			if self.logToConsole:
				plt.show()
				plt.clf()

		return [error_cost, tAccuracy]


	def rebuildEmbeddings(self):
		# find images
		images = []
		for direc, _, files in os.walk(self.dataPath + self.datasetDirectory):
			for file in files:
				if file.endswith("jpg") or file.endswith("jpg"):
					images.append(os.path.join(direc,file))

		# encode
		self.embeddings = {}
		runNumber = {}; # Used to increment the number of the image we are training on (I.E. image ONE... image TWO...)
		for imagePath in tqdm(images, desc="Building embeddings", colour="green", ascii=True, leave=False):
			image = self.encoder.loadImageFile(imagePath) # Load image (this actually cache's the face inside the encoder.)

			name = os.path.basename(os.path.dirname(imagePath)) # the person's name is the name of the folder where the image comes from
			
			faceLocations = self.encoder.faceLocations()
			numberOfFaces = len(faceLocations) # Number of faces?
			#print(number_of_faces)
			if (numberOfFaces==1):
				Id=name.split('.')[0]
				only_name=name.split('.')[1]

				if (Id in runNumber):
					runNumber[Id] = runNumber[Id]+1 # Increment (if key exists)
				else:
					runNumber[Id] = 0 # Create employee image number (key)

				encodingKey = Id + ":" + str(runNumber[Id]) # So each image is under a unique key (ID:run)
				encoding = self.encoder.encode(faceLocations = faceLocations)[0] # Get 128d encoding vector
				self.embeddings[encodingKey] = encoding # Append new encoding under ID:run

			# elif (numberOfFaces > 1):
				# print("Image " + imagePath + " has too many faces. (name: " + name + ")")
			# else: #(numberOfFaces==0):
				# print("Image " + imagePath + " has no faces (name: " + name + ")")
				
		# save
		with open(self.dataPath + self.encodingFile, 'wb') as f:
			pickle.dump(self.embeddings, f)

		self.encoder.cleanUp()

	def rebuild(self, rebuildEmbeddings = True, rebuildVisionNet = True):
		'''
			This will rebuild both the embedding file and the visionNet model.
		'''
		if rebuildEmbeddings:
			self.rebuildEmbeddings()
		if rebuildVisionNet:
			self.rebuildVisionNet()