import numpy as np # ...
# import cupy as np # Numpy in GPU, questionable
from neuralNetwork import genericNeuralNetwork

import matplotlib.pyplot as plt # for plotting
import pandas as pd # logging loss/accuracy values to csv
import pickle # loading encodings
from tqdm import tqdm # loading bar

from copy import deepcopy # evaluate model during training by making exact copy
import os # filesystem, check/make dir


# The number of employees (unique encodings) determines the number of classifiers the network will have.  

# First we need to load our values (the encodings)
print("Loading encodings...")
with open('output/dataset_faces.dat', 'rb') as f:
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

feature_set_builder = None
labels = None
numberOfEncodings = 0
numberOfClassifications = 0
for key in face_encodings:
	if feature_set_builder is None:
		feature_set_builder = np.array(face_encodings[key])
		numberOfEncodings = len(face_encodings[key]) + numberOfEncodings
		numberOfClassifications = numberOfClassifications + 1
		labels = np.array([int(key)-1]*len(face_encodings[key]))
		# labels = np.concatenate((labels, )) # Add 'number' to the labels array x times, where the number we're adding is the employee's ID (the label) and x times is the number of encodings we have for that employee.
	else:
		feature_set_builder = np.concatenate((feature_set_builder, np.array(face_encodings[key])))
		numberOfEncodings = len(face_encodings[key]) + numberOfEncodings
		numberOfClassifications = numberOfClassifications + 1
		labels = np.concatenate((labels, np.array([int(key)-1]*len(face_encodings[key]))))

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

print("\nData loaded and setup completed.")


def ga(model, data, output):
	acc = 0
	for xx,yy in zip(data, output):
		s = model.predictRaw(np.array([xx])).argmax()
		if s == np.argmax(yy):
			acc+=1

	return acc/len(data)*100

if (not os.path.isdir("output/")):
	os.makedirs("output/")
if (not os.path.isdir("output/model_data/")):
	os.makedirs("output/model_data/")

# https://stats.stackexchange.com/questions/181/how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw#:~:text=2/3%20the%20size%20of%20the%20input
# modelGeneric = genericNeuralNetwork(np.array(feature_set), one_hot_labels, hiddenLayers = 0, neurons = int((256/3)+numberOfClassifications), numpyLibrary=np)
modelGeneric = genericNeuralNetwork(np.array(feature_set), one_hot_labels, hiddenLayers = 0, neurons = 6, numpyLibrary=np)
# modelGeneric = genericNeuralNetwork()

#found 50000 iterations with 32 neurons works well

def train(m, epochs = 50000, log = True, logInterval = 100, logAccuracy = True):
	error_cost = []
	tAccuracy = []
	col_names = ['Loss Function','Training accuracy']
	logCSV = pd.DataFrame(columns = col_names)
	print("Training model. (" + str(epochs) + " epochs)")
	m.printInfo()
	for x in tqdm(range(epochs)):
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
					tAccuracy.append(acc)
				logCSV.loc[len(logCSV)] = [loss, acc]

	print("Training accuracy: ", ga(m, feature_set, one_hot_labels))
	
	fileName = str(m.attributes) + "-input_" + str(m.numberOfHiddenLayers+2) + "-layer_" + str(m.neurons) + "-neurons_" + str(m.output_labels) + "-classifications_model"

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
		ax.plot(np.arange(1,len(error_cost)+1), error_cost, color="blue")
		ax.set_ylabel("Loss", color="blue")

		if logAccuracy:
			ax2 = plt.twinx()
			ax2.plot(np.arange(1,len(tAccuracy)+1), tAccuracy, color="orange")
			# ax2.plot(np.arange(1,len(vAccuracy)+1), vAccuracy, color="brown")
			ax2.set_ylabel("Accuracy")
		
		plt.savefig("output/model_data/" + fileName + ".svg", )
		plt.show()
		plt.clf()
	return [error_cost, tAccuracy]

# modelGeneric.loadFile("output/model.ognn")
# modelGeneric.save("output/model.json", fileType = 1)

train(modelGeneric, 50000, logInterval = 1000)
# train(modelGeneric, 1000000, log=False)

print("Training accuracy: ", ga(modelGeneric, feature_set, one_hot_labels))
print(modelGeneric.predictRaw([feature_set[5]]))