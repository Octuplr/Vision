import sys
import os
import numpy as np
import pickle
import cv2
import csv
from PIL import Image, ImageTk
from tqdm import tqdm
import face_recognition # This loads the models with it and takes a considerable amount of time.

# Quality of life
from PIL import ImageDraw, ImageFont
from tkinter import filedialog
import pandas as pd
import datetime
import time

ts = time.time() # This is the current time. Do not penalize the associate for the system processing time.


# https://github.com/ageitgey/face_recognition

faceLocationModelTraining = "hog" # Due to memory limitations, I have to run "hog" when training, but I can use "cnn" during punches.
faceLocationModel = "cnn" # Which face detection model to use. "hog" is less accurate but faster on CPUs. "cnn" is a more accurate deep-learning model which is GPU/CUDA accelerated (if available). Use "cnn" (this consumes MUCH more GPU memory)

faceEncodingsJittersTraining = 100 # Only applied for training. Since I cannot run cnn location model during training (GPU memory limitations) I have to crank this higher to make up for it. Additionally this yields better results in the end.
faceEncodingsJitters = 10 # How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (100 is 100x slower).
# 5 works fine.
# https://face-recognition.readthedocs.io/en/latest/face_recognition.html#:~:text=already%20know%20them.-,num_jitters,-%E2%80%93%20How%20many%20times

faceEncodingsModel = "large"
faceEncodingsModelTrain = "large"

fontsize = 20
font = ImageFont.truetype("arial.ttf", fontsize)

def process_and_encode(images):
    # initialize the list of known encodings and known names
    known_encodings = {}
    ppl_with_bad_imgs = []
    #known_names = []
    print("Encoding faces ...")

    with open('output/AssociateData/AssociateIDMapping.csv','w', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(['Id', 'Name'])
    csvFile.close()

    run = 0;
    for image_path in tqdm(images):
        # Load image
        image = face_recognition.load_image_file(image_path)

        # the person's name is the name of the folder where the image comes from
        name = image_path.split(os.path.sep)[-2]

        #check if there are any faces first
        number_of_faces = len(face_recognition.face_locations(image, model=faceLocationModelTraining))
        #print(number_of_faces)
        if (number_of_faces==1):
            # Encode the face into a 128-d embeddings vector
            
            Id=name.split('.')[0]
            only_name=name.split('.')[1]

            # print("\n" + only_name)
            # below is a hacky way of including every image, it's a horrible idea because it'll make every image its own "cluster" but it's honest work
            known_encodings[Id + ":" + str(run)] = face_recognition.face_encodings(image, num_jitters=faceEncodingsJittersTraining, model=faceEncodingsModelTrain)[0]

            row = [Id + ":" + str(run) , only_name]
            # print(row)
            with open('output/AssociateData/AssociateIDMapping.csv','a', newline='') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()

            run=run+1

        elif(number_of_faces==0):
            ppl_with_bad_imgs.append(name)
            #print("image to change for " + name)
            #print(ppl_with_bad_imgs)
            
    if (len(ppl_with_bad_imgs)):
        # Converting integer list to string list 
        s = [i for i in ppl_with_bad_imgs] 
      
        # Join list items using join() 
        res = " ".join(s)
      
        res = "invalid pics for: " + res
    else:
        res = "Traning finished"
    
    print(res)

    
    return  known_encodings

#######


def TrainImages():
    global known_face_encodings
    global known_face_names
    print("Starting training...")
    
    dataset_dir="data/AssociatePhotos"
    #names,Ids = getNamesAndIds(dataset_dir)
    
    images = []
    for direc, _, files in tqdm(os.walk(dataset_dir)):
        for file in files:
            if file.endswith("jpg"):
                images.append(os.path.join(direc,file))

    #print(images)
    
    known_face_encodings = process_and_encode(images)

    with open('output/dataset_faces.dat', 'wb') as f:
        pickle.dump(known_face_encodings, f)



# def TrainLastImage(imgpath,_face_file_name):

#     # Load face encodings
#     with open('dataset_faces.dat', 'rb') as f:
#         all_face_encodings = pickle.load(f)

#     # Grab the list of names and the list of encodings
#     known_face_names = list(all_face_encodings.keys())
#     known_face_encodings = np.array(list(all_face_encodings.values()))
#     print(os.path.join(imgpath,_face_file_name))
#     image = face_recognition.load_image_file(os.path.join(imgpath,_face_file_name))
#     temp=os.path.join(imgpath,_face_file_name)
#     _name = os.path.basename(temp.split(os.path.sep)[-2])
#     print("name %s : ",_name)
#     print(known_face_encodings.shape)
#     known_face_encodings2=dict(zip(known_face_names, known_face_encodings))
#     print(known_face_encodings2.keys())
#     print("lenght1 %d : ",len(known_face_encodings2))
#     known_face_encodings2[_name]=face_recognition.face_encodings(image)[0]
#     print(known_face_encodings2.keys())
#     #known_face_encodings.append(face_recognition.face_encodings(image)[0])
#     #known_face_names.append(_name)
#     #known_face_encodings=np.concatenate(known_face_encodings,face_recognition.face_encodings(image)[0])


#     print("lenght2 %d : ",len(known_face_encodings2))
#     with open('dataset_faces.dat', 'wb') as f:
#         pickle.dump(known_face_encodings2, f)



def getNamesAndIds(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    print(imagePaths)

    #create empth namelist
    names=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #getting the Id from the image
         # print(os.path.basename(os.path.normpath(imagePath)))
        imagePath_lastFolder=os.path.basename(os.path.normpath(imagePath))
        Id=int(imagePath_lastFolder.split('.')[0])
         # print(imagePath_lastFolder.split('.')[0])

        # extract the face name 
        name=imagePath_lastFolder.split('.')[1]
        names.append(name)
        Ids.append(Id)        
    return names,Ids


def TrackImages(fileName = None, showResult = True):
    col_names =  ['Id','Date','Time','Accuracy']
    attendance = pd.DataFrame(columns = col_names)  
    
    # Load face encodings
    with open('output/dataset_faces.dat', 'rb') as f:
        all_face_encodings = pickle.load(f)     # Load the model

    # Grab the list of names and the list of encodings
    known_face_names = list(all_face_encodings.keys()) # Get employee ID's from model
    known_face_encodings = np.array(list(all_face_encodings.values()))  # Get the encodings for each employee


    if fileName is None:
        loaded_dir_un = filedialog.askopenfilename()
    else:
        loaded_dir_un = fileName
    unknown_image = face_recognition.load_image_file(loaded_dir_un) # Load unknown face

    face_locations = face_recognition.face_locations(unknown_image, model=faceLocationModel) # Find the faces
    unknown_face_encodings = face_recognition.face_encodings(unknown_image, face_locations, num_jitters=faceEncodingsJitters, model=faceEncodingsModel)
    # ^^ Generate the encodings for the unknown image
    

    count_unknown=0
    
    bestAccuracy = 0
    name = "Unknown"

    cords = {}
    for (top, right, bottom, left), unknown_face_encoding in zip(face_locations, unknown_face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding,tolerance=0.75) # Lower tolerance is more strict, (so reverse of accuracy %) .6 is typical best performance
        # ^^ Outputs array of _known_ faces inside the unknown image
        # (high level call, gets the face distances, then gets the names associated with those distances, etc)

        face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)
        # ^^ Gets the actual distances between the unknown faces and the known face clusters
        # (this is used to calculate the best result in addition to the accuracy %)
        accuracyNpFloat = round((1-np.min(face_distances))*100, 2)
        if (accuracyNpFloat>bestAccuracy):
            bestAccuracy = accuracyNpFloat
            cords["left"] = left
            cords["right"] = right
            cords["bottom"] = bottom
            cords["top"] = top
        
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index].split(":")[0] # Index is ID:run, where run is an arbitrary number indicating when this image was trained

        pil_image = Image.fromarray(unknown_image)
        draw = ImageDraw.Draw(pil_image)

        draw.rectangle(((left, top), (right, bottom)), outline=(255, 0, 50))
        text_width, text_height = draw.textsize(name,font=font)
        draw.text((left + 3, bottom - text_height - 5), name + " (" + str(accuracyNpFloat) + ")%", fill=(150, 150, 150, 150))


    #save in file
    if(name!="Unknown"):
        # Draw a box around the face using the Pillow module
        draw.rectangle(((cords["left"], cords["top"]), (cords["right"], cords["bottom"])), outline=(0, 0, 255))

        # font = getFontSize(name,top, right, bottom, left)
        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name,font=font)
        draw.rectangle(((cords["left"], cords["bottom"] - text_height - 10), (cords["right"], cords["bottom"])), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((cords["left"] + 3, cords["bottom"] - text_height - 5), name + " (" + str(bestAccuracy) + ")%", fill=(255, 255, 255, 255))
        date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        # Id=name.split('.')[0]
        # only_name=name.split('.')[1]
        acry = str(bestAccuracy/100)
        attendance.loc[len(attendance)] = [name,date,timeStamp,acry]
    else :
        print("Unknown person.")
        if (showResult == True):
            pil_image.show()
        return
            
    
    
    # create attendance file   
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="output/Punches/Punch_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    img_fileName="output/Punches/Punch_"+date+"_"+Hour+"-"+Minute+"-"+Second+".png"
    pil_image.save(img_fileName)
    attendance.to_csv(fileName,index=False)
    print("Punch (ID): " + name + " (" + str(bestAccuracy) + "%)");

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    if (showResult == True):
        pil_image.show()



if __name__ == '__main__':
    args = sys.argv
    argc = len(args)

    needsHelp = False

    if (not os.path.isdir("output/")):
        os.makedirs("output/")
    if (not os.path.isdir("output/AssociateData/")):
        os.makedirs("output/AssociateData/")
    if (not os.path.isdir("output/Punches/")):
        os.makedirs("output/Punches/")

    if (argc < 2):
        needsHelp = True
    elif (args[1] == "-train"):
        TrainImages()
        quit()
    elif (args[1] == "-h" or args[1] == "-help" or args[1] == "-?"):
        needsHelp = True
    elif (args[1] == "-punch"):
        if (argc >= 3):
            if (args[2] == "-noshow" or args[2] == "-ns"):
                TrackImages(showResult = False)
            else:
                TrackImages(args[2])
            quit();
        else:
            TrackImages()
            quit();

    elif (argc >= 3):
        if (args[1] == "append"):
            print("[WIP] = Not yet implemented.")
            quit()

    print('Usage: %s\n' % args[0]
        + '\t-train: Train model for ALL associate faces.\n'
        + '\t-append <associate ID>: Adds an associates face data to the model.\n'
        + '\t-punch [path-to-picture]: Checks an image for an associate. (If no image provided, a file picker dialog will open).');
    quit()