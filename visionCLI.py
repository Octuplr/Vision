print("loading visionAPI...")
from visionAPI import Vision
import sys
import os
from tkinter import filedialog
import numpy as np

def punch(filePath = None, showResult = False):
	if filePath is None:
		filePath = filedialog.askopenfilename(title = "Select associate photo", defaultextension = "jpg", initialdir = os.getcwd() + "/data/TestingPhotos/")
	elif not os.path.isfile(filePath):
		filePath = filedialog.askopenfilename(title = "Select associate photo", defaultextension = "jpg", initialdir = os.getcwd() + "/data/TestingPhotos/")

	associates, confidences, img = vision.addAssociatePunch(filePath, True, drawOnImage = True)

	# for employeeID, confidence in zip(associates, confidences):
		# print("Punch for ID " + str(employeeID) + " (gnn confidence: " + str(confidence) + "%)")

	if showResult:
		img.show()
		del img



vision = Vision(dataPath = "output/", datasetDirectory = "../data/AssociatePhotos/", logToConsole = True)

clear = "clear"
if os.name == 'nt':
	clear = "cls"

# Screen helpers
def clearScreen():
	print("\x1B[0m\x1B[J\x1B[H")
def clearLine():
	print("\x1B[2K")
def ansiSGR(msg, SGRCode):
	return "\x1B[" + str(SGRCode) + "m" + msg + "\x1B[0m"
def uL(msg):
	return ansiSGR(msg, "4")
def b(msg):
	return ansiSGR(msg, "11")
def ansiControl(command, n, m = None):
	if m is None:
		print("\x1B[" + str(n) + ";" + "\x1B[" + command)
	else:
		print("\x1B[" + str(n) + ";" + str(m) + "\x1B[" + command)
def cursorUp():
	ansiControl('A', 1)
def cursorDown():
	ansiControl('B', 1)
def cursorForward():
	ansiControl('C', 1)
def cursorBack():
	ansiControl('D', 1)
def printHeader():
	clearScreen()
	print("=== Vision CLI ===")
	print("1) Clock associate in")
	print("2) Rebuild databases (embeddings and VisionNet model)")
	print("3) Rebuild face embeddings")
	print("4) Rebuild VisionNet model")
	print("5) Quit")
	print()
	print("6) Test punches")
	print("7) Test accuracy?")

def enterToReturn():
	cursorDown()
	print(ansiSGR("PPress enter to return", "90"))
	input()


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

if __name__ == '__main__':
	args = sys.argv
	argc = len(args)

	needsHelp = False


	if (not os.path.isdir("output/")):
		os.makedirs("output/")
	if (not os.path.isdir("output/Punches/")):
		os.makedirs("output/Punches/")

	if (argc < 2):
		bRun = True
		os.system(clear)
		while bRun:
			# clear
			printHeader()
			clearLine()
			option = int(input("Command? "))

			print()
			print()

			if option == 1:
				punch(None, True)
				enterToReturn()
				os.system(clear)
			elif option == 2:
				if str(input("Are you sure (Y/N)? ")).lower().startswith("y"):
					vision.rebuild() # okay I guess
				print("Embeddings rebuilt")
				enterToReturn()
				os.system(clear)
			elif option == 3:
				if str(input("Are you sure (Y/N)? ")).lower().startswith("y"):
					vision.rebuildEmbeddings() # okay I guess
				print("VisionNet rebuilt")
				enterToReturn()
				os.system(clear)
			elif option == 4:
				if str(input("Are you sure (Y/N)? ")).lower().startswith("y"):
					vision.rebuildVisionNet(log=True) # okay I guess
				print("Embeddings and VisionNet rebuilt")
				enterToReturn()
				os.system(clear)
			elif option == 5:
				bRun = False
				break

			elif option == 6:
				while True:
					punch(None, True)

			elif option == 7:
				feature_set, one_hot_labels, idMappings = vision.loadEncodings('output/testing-encodings.dat')
				testing = ga(vision.neuralNetwork, feature_set, one_hot_labels)
				print("Testing correctness:  ", testing[0])
				print("Testing avg accuracy: ", testing[1])
		
		needsHelp = True
		os.system(clear)

	elif (args[1] == "-train"):
		vision.rebuild()
		quit()
	elif (args[1] == "-h" or args[1] == "-help" or args[1] == "-?"):
		needsHelp = True
	elif (args[1] == "-punch"):
		if (argc >= 3):
			if (args[2] == "-noshow" or args[2] == "-ns"):
				punch(showResult = False)
			else:
				punch(args[2])
			quit();
		else:
			punch()
			quit();

	elif (argc == 3 and (args[1] == "-a" or args[1] == "-append")):
		AppendEncoding(args[2])
		quit();

	elif (argc >= 4):
		if (args[1] == "-append" or args[1] == "-a"):
			AppendEncoding(args[2], args[3])
			quit();

	print('Usage: %s\n' % args[0]
		+ '\t-train: Regenerate all face embeddings and retrain the VisionNet model.\n'
		# + '\t-append | -a: <associate ID> [path-to-picture]: Adds an associates face data to the model.\n'
		+ '\t-punch [path-to-picture]: Checks an image for an associate. (If no image provided, a file picker dialog will open).');
	quit()