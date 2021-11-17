import numpy as np # ...
# import cupy as np # Numpy in GPU, questionable
import matplotlib.pyplot as plt # for plotting
import pickle # loading encodings

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

print("Data loaded and setup completed.")


def sigmoid(x): # Squish input to (0,1)
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x) *(1-sigmoid (x))
def sigmoid_derv(s):
	return s * (1 - s)

def softmax(A):
	exps = np.exp(A - np.max(A, axis=1, keepdims=True))
	return exps/np.sum(exps, axis=1, keepdims=True)
    # expA = np.exp(A)
    # return expA / expA.sum(axis=1, keepdims=True)


def cross_entropy(pred, real):
    n_samples = real.shape[0]
    res = pred - real
    return res/n_samples

def error(pred, real):
	n_samples = real.shape[0]
	logp = - np.log(pred[np.arange(n_samples), real.argmax(axis=1)])
	loss = np.sum(logp)/n_samples
	return loss



class Nision3LayerNN:
	def __init__(self, x, y):
		self.input = x
		self.output = y
		self.lr = 0.005			# Learning rate
		neurons = 128				# Number of hidden nodes
		attributes = x.shape[1]		# Number of inputs
		output_labels = y.shape[1]	# Number of possible classifications

		# We have two hidden layers, this requires three weights/biases
		#
		#  input 			a1			a2		   a3
		#	(O)		 __    (O)	 __    (O)	 __   (O)
		#	(O)		/  |   (O)	|  |   (O)	|  \  (O)
		#	(O)		|  |   (O)	|  |   (O)	|  |  (O)
		#	 .		|W1|    .	|W2|    .	|W3|   .
		#	 .		|  |    .	|  |    .	|  |   .
		#	(O)		\__|   (O)	|__|   (O)	|__/  (O)
		#  n=128 	 |	  n=128  |	  n=128  |	  n=N
		#	    (128 x neurons)  |   (neurons x output_labels)
		#				(neurons x neurons)
		#

		self.w1 = np.random.rand(attributes, neurons)	# Weights	(1) (input_size, output_size)
		self.b1 = np.random.randn(neurons) 				# Biases	(1)
		self.w2 = np.random.rand(neurons, neurons) 		# Weights	(2)
		self.b2 = np.random.randn(neurons) 				# Biases	(2)
		self.w3 = np.random.rand(neurons, output_labels)# Weights	(3)
		self.b3 = np.random.randn(output_labels)		# Biases	(3)

	def feedforward(self):
		z1 = np.dot(self.input, self.w1) + self.b1	# z1
		self.a1 = sigmoid(z1)						# Activation function for layer 1
		z2 = np.dot(self.a1, self.w2) + self.b2		# z2
		self.a2 = sigmoid(z2)						# Activation function for layer 2
		z3 = np.dot(self.a2, self.w3) + self.b3		# z3
		self.a3 = softmax(z3)						# Activation function for layer 3 (out of the neural network)


	def backprop(self):
		a3_delta = self.a3 - one_hot_labels #cross_entropy(self.a3, self.output) # w3
		z2_delta = np.dot(a3_delta, self.w3.T)
		a2_delta = z2_delta * sigmoid_derv(self.a2) # w2
		z1_delta = np.dot(a2_delta, self.w2.T)
		a1_delta = z1_delta * sigmoid_derv(self.a1) # w1

		self.w3 -= self.lr * np.dot(self.a2.T, a3_delta)
		self.b3 -= self.lr * np.sum(a3_delta, axis=0)
		self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
		self.b2 -= self.lr * np.sum(a2_delta, axis=0)
		self.w1 -= self.lr * np.dot(self.input.T, a1_delta)
		self.b1 -= self.lr * np.sum(a1_delta, axis=0)

	def predict(self, data):
		self.input = data
		self.feedforward()
		return self.a3.argmax()

	def printLoss(self):
		loss = error(self.a3, self.output)
		print('Loss function value: ', loss)


class Nision2LayerNN:
	def __init__(self, x, y):
		self.input = x
		self.output = y
		self.lr = 0.001			# Learning rate
		neurons = 128				# Number of hidden nodes
		attributes = x.shape[1]		# Number of inputs
		output_labels = y.shape[1]	# Number of possible classifications

		# We have two hidden layers, this requires three weights/biases
		#
		#  input 			a1 		   a2
		#	(O)		 __    (O)	 __   (O)
		#	(O)		/  |   (O)	|  \  (O)
		#	(O)		|  |   (O)	|  |  (O)
		#	 .		|W1|    .	|W2|   .
		#	 .		|  |    .	|  |   .
		#	(O)		\__|   (O)	|__/  (O)
		#  n=128 	 |	  n=128  |	  n=N
		#	    (128 x neurons)	 |
		#				(neurons x output_labels)
		#

		self.w1 = np.random.rand(attributes, neurons)	# Weights	(1) (input_size, output_size)
		self.b1 = np.random.randn(neurons) 				# Biases	(1)
		self.w2 = np.random.rand(neurons, output_labels)# Weights	(2)
		self.b2 = np.random.randn(output_labels)		# Biases	(2)

	def feedforward(self):
		z1 = np.dot(self.input, self.w1) + self.b1	# z1
		self.a1 = sigmoid(z1)						# Activation function for layer 1
		z2 = np.dot(self.a1, self.w2) + self.b2		# z2
		self.a2 = softmax(z2)						# Activation function for layer 2


	def backprop(self):
		a2_delta = self.a2 - one_hot_labels #cross_entropy(self.a3, self.output) # w3
		z1_delta = np.dot(a2_delta, self.w2.T)
		a1_delta = z1_delta * sigmoid_derv(self.a1) # w1

		self.w2 -= self.lr * np.dot(self.a1.T, a2_delta)
		self.b2 -= self.lr * np.sum(a2_delta, axis=0)
		self.w1 -= self.lr * np.dot(self.input.T, a1_delta)
		self.b1 -= self.lr * np.sum(a1_delta, axis=0)

	def predict(self, data):
		self.input = data
		self.feedforward()
		return self.a2.argmax()

	def printLoss(self):
		loss = error(self.a2, self.output)
		print('Loss function value: ', loss)


# model3 = Nision3LayerNN(feature_set, one_hot_labels)

# epochs = 50000
# for x in range(epochs):
# 	model3.feedforward()
# 	model3.backprop()
# 	if x % 2000 == 0:
# 		model3.printLoss()

def ga(model, data, output):
	acc = 0
	for xx,yy in zip(data, output):
		s = model.predict([xx])
		if s == np.argmax(yy):
			print("Right")
			acc+=1
	return acc/len(data)*100

# print("Training accuracy: ", ga(model3, feature_set, [one_hot_labels]))
# print("")
# print("")
# print("")

model2 = Nision2LayerNN(feature_set, one_hot_labels)

epochs = 5000000
for x in range(epochs):
	model2.feedforward()
	model2.backprop()
	if x % 2000 == 0:
		model2.printLoss()

print("Training accuracy: ", ga(model2, feature_set, [one_hot_labels]))
print("")
print("")
print("")