import numpy as npA # ...
# import cupy as np # Numpy in GPU, questionable
np = npA

from neuralNetwork import genericNeuralNetwork

import matplotlib.pyplot as plt # for plotting
import pandas as pd # logging loss/accuracy values to csv
import pickle # loading encodings
from tqdm import tqdm # loading bar

from copy import deepcopy # evaluate model during training by making exact copy
import os # filesystem, check/make dir


# The number of employees (unique encodings) determines the number of classifiers the network will have.  

# First we need to load our values (the encodings)
numberOfClassifications = 0
def loadEncodings(location):
	print("Loading encodings...")
	with open(location, 'rb') as f:
		known_face_encodings = pickle.load(f)

	face_encodings = {}

	for key in known_face_encodings:
	    encodingId = key.split(":")[0]
	    if encodingId in face_encodings:
	    	face_encodings[encodingId].append(known_face_encodings[key]) # Append
	    else:
	    	face_encodings[encodingId] = [known_face_encodings[key]]

	print("Building feature set...")

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
	global numberOfClassifications
	for key in face_encodings:
		if feature_set_builder is None:
			feature_set_builder = np.array(face_encodings[key])
			numberOfEncodings = len(face_encodings[key]) + numberOfEncodings
			numberOfClassifications = numberOfClassifications + 1
			labels = np.array([idMappingsReversed[key]]*len(face_encodings[key]))
			# labels = np.concatenate((labels, )) # Add 'number' to the labels array x times, where the number we're adding is the employee's ID (the label) and x times is the number of encodings we have for that employee.
		else:
			feature_set_builder = np.concatenate((feature_set_builder, np.array(face_encodings[key])))
			numberOfEncodings = len(face_encodings[key]) + numberOfEncodings
			numberOfClassifications = numberOfClassifications + 1
			labels = np.concatenate((labels, np.array([idMappingsReversed[key]]*len(face_encodings[key]))))

	feature_set = np.vstack(feature_set_builder)
	print("Data labels: ")
	print(labels)


	# Build output label
	one_hot_labels = np.zeros((numberOfEncodings, numberOfClassifications))

	for i in range(numberOfEncodings):
	    one_hot_labels[i, labels[i]] = 1

	# Show the plot
	# plt.scatter(feature_set[:,0], feature_set[:,1], c=labels, cmap='plasma', s=100, alpha=0.5)
	# plt.show()

	print("Data loaded and setup completed.\n")
	return feature_set, one_hot_labels, idMappings


# Returns: [correctness, accuracy, [correctLabel, predictionLabel, accuracy]]
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

if (not os.path.isdir("output/")):
	os.makedirs("output/")
if (not os.path.isdir("output/model_data/")):
	os.makedirs("output/model_data/")



# feature_set, one_hot_labels, idMappings = loadEncodings('output/testing-encodings.dat')
feature_set, one_hot_labels, idMappings = loadEncodings('output/encodings.dat')


# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw#:~:text=2/3%20the%20size%20of%20the%20input
# modelGeneric = genericNeuralNetwork(np.array(feature_set), one_hot_labels, hiddenLayers = 0, neurons = int((256/3)+numberOfClassifications), numpyLibrary=np, implementationData=idMappings)
modelGeneric = genericNeuralNetwork(np.array(feature_set), one_hot_labels, hiddenLayers = 0, neurons = int(numberOfClassifications*8), numpyLibrary=np, implementationData=idMappings)
# modelGeneric = genericNeuralNetwork()

#found 50000 iterations with 32 neurons works well

def train(m, epochs = 50000, log = True, logInterval = 100, logAccuracy = True):
	# Epochs: number of cycles of training
	# Log: Log both the loss function and training accuracy/correctness
	error_cost = []
	tAccuracy = []
	col_names = ['Loss Function','Training correctness','Average training accuracy']
	logCSV = pd.DataFrame(columns = col_names)
	
	m.printInfo()
	print("\n")
	for x in tqdm(range(epochs), desc="Training model", colour="green", ascii=True):
	# for x in range(epochs):
		m.feedforward()
		m.backprop()
		if log:
			if x % logInterval == 0:
				loss = m.getLoss()
				error_cost.append(loss)
				acc = None
				if logAccuracy == True:
					acc = ga(deepcopy(m), feature_set, one_hot_labels)
					tAccuracy.append(acc[1])
				logCSV.loc[len(logCSV)] = [loss, acc[0], acc[1]]

	acc = ga(m, feature_set, one_hot_labels)
	print("Training correctness:", acc[0])
	print("Average accuracy:\t", acc[1])
	
	fileName = str(m.attributes) + "-input_" + str(m.numberOfHiddenLayers+2) + "-layer_" + str(m.neurons) + "-neurons_" + str(m.output_labels) + "-classifications_model-ID_" + m.modelID

	# Save model
	modelGeneric.save("output/model_data/" + fileName + ".ognn") # Saves without activations or input data, but with the output data?

	# Log to CSV
	if log:
		logCSV.to_csv("output/model_data/" + fileName + ".csv",index=False)

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
		
		plt.savefig("output/model_data/" + fileName + ".svg", )
		plt.show()
		plt.clf()
	return [error_cost, tAccuracy]

# modelGeneric.loadFile("output/model.ognn")
# modelGeneric.save("output/model.json", fileType = 1)

train(modelGeneric, 50000, logInterval = 1000)
# train(modelGeneric, 500000, log=False)


# Evaluate model
feature_set, one_hot_labels, idMappings = loadEncodings('output/testing-encodings.dat')
testing = ga(modelGeneric, feature_set, one_hot_labels)
print("Testing correctness:  ", testing[0])
print("Testing avg accuracy: ", testing[1])
# print(modelGeneric.predictRaw([feature_set[5]]))