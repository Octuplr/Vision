import numpy as np
import cupy as cp # Numpy in GPU, questionable

# for saving/exporting model
from copy import deepcopy # create export
import pickle # save ("raw") to file
import json # save as json file
from json import JSONEncoder # encode ndarray
import io

# Generating random ID
import random
import string

def sigmoid(x): # Squish input to (0,1)
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))
def sigmoid_derv(s): # Obv. faster, questionable if improves accuracy.
    # return sigmoid(s) *(1-sigmoid (s))
	return s * (1 - s)

def softmax(A):
	exps = np.exp(A - np.max(A, axis=1, keepdims=True))
	return exps/np.sum(exps, axis=1, keepdims=True)
    # expA = np.exp(A)
    # return expA / expA.sum(axis=1, keepdims=True)

def error(pred, real):
	n_samples = real.shape[0]
	logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
	loss = np.sum(logp)/n_samples
	return loss

# encode ndarray (saving)
class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class genericNeuralNetwork:
	def __init__(self, data=np.zeros(shape=(1,1)), output=np.zeros(shape=(1,1)), attributes=0, outputs=0, learningRate = 0.005, hiddenLayers = 0, neurons = 0, numpyLibrary = np, implementationData = None, hashData = True, modelID = None):
		'''
		Builds a feedforward neural network
		:param data: Array of the training data [ [input1, input2, ...], [input1, input2, ...], ..., [input1, input2, ...]]
		:param output: Array of one-hot vectors
		:param attributes: Number of inputs
		:param outputs: Number of classifications (...outputs)
		:param learningRate: Learning rate
		:param hiddenLayers: Number of layers + 2 (input and output)
		:param neurons: Number of interlinking neurons between the layers
		:param numpyLibrary: The numpy library we use (so you can use cupy or numpy)
		:param implementationData: Additional to store with the model
		:param hashData: When building a unique model ID we hash the input data to better help distinguish models from each other. Setting this to false means we do not hash the input data.
		:param modelID: This is the unique identifier for the model. If you provide a string, we set the modelID to it. Otherwise it is set to: hash({attributes}_{hiddenLayers+2}_{neurons}_{outputs}?_{hash(data)}) (where the end is either random or the data array hashed.)



		'''
		np = numpyLibrary
		if (np == cp):
			self.library = "cupy"
			self.input = np.array(data)
			self.output = np.array(output)
		else:
			self.library = "numpy"
			self.input = data
			self.output = output

		self.lr = learningRate		# Learning rate
		self.neurons = neurons
		if (self.neurons <= 0): 	# Number of neurons has to be more than the number of inputs
			self.neurons = data.shape[1]	# Number of neurons = number of inputs
		if (hiddenLayers<0):
			hiddenLayers = 0

		self.attributes = attributes	# Number of inputs
		if (self.attributes <= 0):
			self.attributes = data.shape[1]

		self.output_labels = outputs 	# Number of possible classifications (outputs)
		if (self.output_labels <= 0):
			self.output_labels = output.shape[1]	

		self.iterations = 0

		self.implementationData = implementationData

		# Generate a unique ID
		self.modelID = hash(str(self.attributes) + "_" + str(hiddenLayers+2) + "_" + str(self.neurons) + "_" + str(self.output_labels))
		if hashData == True:
			self.modelID = hash(str(self.modelID) + "_".join(random.SystemRandom().choice(string.ascii_uppercase + string.digits) for _ in range(32)))


		# We have n hidden layers, this requires n+2 weights/biases
		#
		#  input 			ai			aN		   ao
		#	(O)		 __    (O)	 __    (O)	 __   (O)
		#	(O)		/  |   (O)	|  |   (O)	|  \  (O)
		#	(O)		|  |   (O)	|  |   (O)	|  |  (O)
		#	 .		|iW|    .	|nW|    .	|oW|   .
		#	 .		|  |    .	|  |    .	|  |   .
		#	(O)		\__|   (O)	|__|   (O)	|__/  (O)
		#  n=attr. 	 | n=neurons | n=neurons |	  n=output_labels
		#			 |			 |			 |
		#(attributes x neurons)  |  (neurons x output_labels)
		#				(neurons x neurons)
		#

		self.iWeight = np.random.rand(self.attributes, neurons)		# Weights	(input) (input_size, output_size)
		self.iBias = np.random.randn(neurons) 						# Biases	(input layer)

		# generate the n hidden layers
		self.numberOfHiddenLayers = hiddenLayers
		self.hLayers = []
		for i in range(self.numberOfHiddenLayers):
			self.hLayers.append({})
			self.hLayers[i]['w'] = np.random.rand(neurons, neurons)	# hidden layer i weights
			self.hLayers[i]['b'] = np.random.randn(neurons)			# hidden layer i biases

		self.oWeight = np.random.rand(neurons, self.output_labels)	# Weights	(output layer)
		self.oBias = np.random.randn(self.output_labels)			# Biases	(output layer)

	def feedforward(self): # Predict output
		# z<layer> = np.dot(leftHandInputs/neurons, <layer_weights>) + <layer_bias>
		# activation<layer> = sigmoid(z<layer>) (the output of this "layer")
		zinput = np.dot(self.input, self.iWeight) + self.iBias
		self.ai = sigmoid(zinput) # Activation function for layer 1 (input)
		
		lastActivation = self.ai
		for i in range(self.numberOfHiddenLayers):
			# self.hLayers[i]['z']
			z = np.dot(lastActivation, self.hLayers[i]['w']) + self.hLayers[i]['b']
			lastActivation = sigmoid(z)
			self.hLayers[i]['a'] = lastActivation

		zout = np.dot(lastActivation, self.oWeight) + self.oBias # z2
		self.ao = softmax(zout)	# Activation function for output layer (model output)


	def backprop(self): # Train weights
		#z<layer>_delta = np.dot(z<layer-1>_delta, (<layer-1>_weights).T)
		#a<layer>_delta = z<layer>_delta * sigmoid_derv(<layer>_activation)

		ao_delta = self.ao - self.output #cross_entropy(self.a3, self.output) # w3
		lastDelta = ao_delta
		lastWeights = self.oWeight.T

		for i in range(self.numberOfHiddenLayers):
			# self.hLayers[i]['zDelta']
			zDelta = np.dot(lastDelta, lastWeights)
			lastDelta = zDelta * sigmoid_derv(self.hLayers[i]['a'])
			self.hLayers[i]['aDelta'] = lastDelta
			lastWeights = self.hLayers[i]['w'].T
		
		z = np.dot(lastDelta, lastWeights)
		ai_delta = z * sigmoid_derv(self.ai) # w1

		# Phase two
		# <layer>_weights -= self.lr * np.dot(<layer-1>_activation.T, <layer>_aDelta) (leftHandInputs/neurons = layer-1_activation)
		# <layer>_bias -= self.lr * np.sum(<layer>_aDelta, axis=0)
		lastActivation = self.ai.T
		for i in range(self.numberOfHiddenLayers):
			self.hLayers[i]['w'] -= self.lr * np.dot(lastActivation, self.hLayers[i]['aDelta'])
			self.hLayers[i]['b'] -= self.lr * np.sum(self.hLayers[i]['aDelta'], axis=0)
			del self.hLayers[i]['aDelta']
			lastActivation = self.hLayers[i]['a'].T
		
		self.oWeight -= self.lr * np.dot(lastActivation, ao_delta)
		self.oBias -= self.lr * np.sum(ao_delta, axis=0)

		self.iWeight -= self.lr * np.dot(self.input.T, ai_delta)
		self.iBias -= self.lr * np.sum(ai_delta, axis=0)

		self.iterations = self.iterations + 1 # Number of weight updates


	def predictRaw(self, data):
		if type(data) is list:
			self.input = np.copy(data)
		else:
			self.input = data
		self.feedforward()
		return self.ao
	def predict(self, data):
		return self.predictRaw(data).argmax()

	def getLoss(self):
		return error(self.ao, self.output)

	def printInfo(self):
		print("Generic Neural Network.")
		print("\tModel ID: " + str(self.modelID))
		print("\tNumber of inputs:  " + str(self.attributes))
		print("\tNumber of layers:  " + str(self.numberOfHiddenLayers+2))
		print("\tNumber of neurons: " + str(self.neurons))
		print("\tNumber of outputs: " + str(self.output_labels))
		print("\tNumber of iterations: " + str(self.iterations))
		# print("\tCurrent loss function: " + str(self.getLoss()))

	def export(self, withOutput = True, withActivations = False, withInputData = False):
		modelData = {}

		# You don't need this if you can supply the proper output array, but why not add it in.
		if withOutput:
			modelData['output'] = deepcopy(self.output)

		# Generic properties (not necessarily needed)
		modelData['neurons'] = self.neurons
		modelData['attributes'] = self.attributes
		modelData['output_labels'] = self.output_labels
		modelData['numberOfHiddenLayers'] = self.numberOfHiddenLayers
		modelData['iterations'] = self.iterations
		modelData['modelID'] = self.modelID
		modelData['library'] = self.library

		#implementationData
		modelData['implementationData'] = deepcopy(self.implementationData)

		# Weights/Biases (the actual model, definitely needed)
		modelData['iWeight'] = deepcopy(self.iWeight)
		modelData['iBias'] = deepcopy(self.iBias)
		modelData['hLayers'] = deepcopy(self.hLayers)
		modelData['oWeight'] = deepcopy(self.oWeight)
		modelData['oBias'] = deepcopy(self.oBias)

		# If you, for some reason, want to save the current data to file. Probably a bad idea
		if withInputData:
			modelData['input'] = deepcopy(self.input)

		# Never needed unless you want to be able to call backprop() after loading ... which is stupid ... but good for saving an exact clone to file
		if withActivations:
			modelData['ao'] = deepcopy(self.ao)
			modelData['ai'] = deepcopy(self.ai)
		else:
			for i in range(self.numberOfHiddenLayers):
				del modelData['hLayers'][i]['a']

		return modelData

	def save(self, filename, fileType = 0, withOutput = True, withActivations = False, withInputData = False):
		# File type 0: pickle.dump to file
		# File type 1: JSON file (but why)
		modelData = self.export(withOutput, withActivations, withInputData)

		if fileType == 0 or fileType == "pickle" or fileType == "raw" or fileType == "ognn":
			with open(filename, 'wb') as f:
				pickle.dump(modelData, f)
		elif fileType == 1 or fileType == "json" or fileType == "JSON" or fileType == "text":
			# why are you doing this
			print(" !!! CAUTION !!! you currently CANNOT load a model from a JSON file. Only use this file type to save the model for demonstration purposes ONLY!")
			print("You have been warned.")
			with open(filename, 'w') as f:
				json.dump(modelData, f, cls=NumpyArrayEncoder)


	def load(self, data, strict = True):
		if "neurons" in data:
			self.neurons = data['neurons']
		else:
			if strict:
				raise MissingModelData("Missing neurons (neuron count)")

		if "attributes" in data:
			self.attributes = data['attributes']
		else:
			if strict:
				raise MissingModelData("Missing attributes (input size)")

		if "output_labels" in data:
			self.output_labels = data['output_labels']
		else:
			if strict:
				raise MissingModelData("Missing output_labels (output size)")

		if "numberOfHiddenLayers" in data:
			self.numberOfHiddenLayers = data['numberOfHiddenLayers']
		else:
			raise MissingModelData("Missing numberOfHiddenLayers (layer count -2)")

		if "iterations" in data:
			self.iterations = data['iterations']
		else:
			if strict:
				raise MissingModelData("Missing iterations (number of weight updates / training iterations)")

		if "library" in data:
			self.library = data['library']


		if "modelID" in data:
			self.modelID = data['modelID']
		else:
			if strict:
				raise MissingModelData("Missing model ID (unique identifier for this model)")

		if "implementationData" in data:
			self.implementationData = data['implementationData']

		
		if "iWeight" in data:
			self.iWeight = np.array(data['iWeight'])
		else:
			raise MissingModelData("Missing iWeight (input weights)")

		if "iBias" in data:
			self.iBias = np.array(data['iBias'])
		else:
			raise MissingModelData("Missing iBias (input biases)")
		if "hLayers" in data:
			self.hLayers = np.array(data['hLayers'])
			for i in range(self.numberOfHiddenLayers):
				self.hLayers[i]['w'] = np.array(self.hLayers[i]['w'])
				self.hLayers[i]['b'] = np.array(self.hLayers[i]['b'])
		else:
			if (self.numberOfHiddenLayers != 0):
				raise MissingModelData("Missing hLayers (hidden layer(s) data)")
		if "oWeight" in data:
			self.oWeight = np.array(data['oWeight'])
		else:
			raise MissingModelData("Missing oWeight (output weights)")
		if "oBias" in data:
			self.oBias = np.array(data['oBias'])
		else:
			raise MissingModelData("Missing oBias (out biases)")


		if "input" in data:
			self.input = np.array(data['input'])

		if "output" in data:
			self.output = np.array(data['output'])

		if "ao" in data and "ai" in data:
			self.ao = np.array(data['ao'])
			self.ai = np.array(data['ai'])
			for i in range(self.numberOfHiddenLayers):
				self.hLayers[i]['a'] = np.array(self.hLayers[i]['a'])

	def loadFile(self, filename, fileType = 0):
		if fileType == 0 or fileType == "pickle" or fileType == "raw" or fileType == "ognn":
			try:
				with open(filename, 'rb') as f:
					self.load(pickle.load(f))
			except io.UnsupportedOperation as exp:
				raise LoadException("No such model file found.")
		else:
			raise LoadException("File type != 0 unsupported. Only \"pickle\" files may be loaded.")

class LoadException(Exception):
	'''raise this when loading an unsupported file (type)'''
class MissingModelData(NameError):
    '''raise this when there's missing data when loading model data'''