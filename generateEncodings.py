import os

from tqdm import tqdm
from faceEncoding import faceEncoder

import pickle

def process_and_encode(images):
    print("Encoding ...")
    embeddings = {}
    
    runNumber = {}; # Used to increment the number of the image we are training on (I.E. image ONE... image TWO...)
    for imagePath in tqdm(images):
        image = encoder.loadImageFile(imagePath) # Load image (this actually cache's the face inside the encoder.)

        name = imagePath.split(os.path.sep)[-2] # the person's name is the name of the folder where the image comes from
        
        faceLocations = encoder.faceLocations()
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
            encoding = encoder.encode(faceLocations = faceLocations)[0] # Get 128d encoding vector
            embeddings[encodingKey] = encoding # Append new encoding under ID:run

        # elif (numberOfFaces > 1):
            # print("Image " + imagePath + " has too many faces. (name: " + name + ")")
        # else: #(numberOfFaces==0):
            # print("Image " + imagePath + " has no faces (name: " + name + ")")
            

    return  embeddings

#######



encoder = faceEncoder(shapeModelSize="large", faceDetectorModel="hog", numJitters=100, numUpsamples=1)

print("Finding images...")
    
dataset_dir="data/AssociatePhotos"
    
images = []
for direc, _, files in os.walk(dataset_dir):
    for file in files:
        if file.endswith("jpg"):
            images.append(os.path.join(direc,file))

embeddings = process_and_encode(images)

print("Saving embeddings (encodings)...")
with open('output/encodings.dat', 'wb') as f:
    pickle.dump(embeddings, f)

print("Complete")