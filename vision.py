import shutil
import face_recognition
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import time
import pickle
from tkinter import filedialog

import tkinter as tk
from tkinter import * 
from tkinter import Message ,Text
import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import tkinter.ttk as ttk
import tkinter.font as font

def process_and_encode(images):
    # initialize the list of known encodings and known names
    known_encodings = {}
    ppl_with_bad_imgs = []
    #known_names = []
    print("Encoding faces ...")

    # with open('output/AssociateData/AssociateIDMapping.csv','w', newline='') as csvFile:
    #     writer = csv.writer(csvFile)
    #     writer.writerow(['Id', 'Name'])
    # csvFile.close()

    for image_path in tqdm(images):
        # Load image
        image = face_recognition.load_image_file(image_path)

        # the person's name is the name of the folder where the image comes from
        name = image_path.split(os.path.sep)[-2]

        #check if there are any faces first
        number_of_faces = len(face_recognition.face_locations(image))
        #print(number_of_faces)
        if (number_of_faces==1):
            # Encode the face into a 128-d embeddings vector
            known_encodings[name] = face_recognition.face_encodings(image)[0]
            print(name)
            # Id=name.split('.')[0]
            # only_name=name.split('.')[1]
            # row = [Id , only_name]
            #print(row)
            # with open('output/AssociateData/AssociateIDMapping.csv','a', newline='') as csvFile:
            #     writer = csv.writer(csvFile)
            #     writer.writerow(row)
            # csvFile.close()

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


def TrackImages():

    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)  
    
    # Load face encodings
    with open('output/dataset_faces.dat', 'rb') as f:
        all_face_encodings = pickle.load(f)

    # Grab the list of names and the list of encodings
    known_face_names = list(all_face_encodings.keys())
    known_face_encodings = np.array(list(all_face_encodings.values()))

    #unknown_image = face_recognition.load_image_file("lecturer_test.jpg")
    loaded_dir_un = filedialog.askopenfilename()
    unknown_image = face_recognition.load_image_file(loaded_dir_un)

    face_locations = face_recognition.face_locations(unknown_image)
    unknown_face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    pil_image = Image.fromarray(unknown_image)
    draw = ImageDraw.Draw(pil_image)

    count_unknown=0
    for (top, right, bottom, left), unknown_face_encoding in zip(face_locations, unknown_face_encodings):
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, unknown_face_encoding,tolerance=0.45)

        name = "Unknown"

        face_distances = face_recognition.face_distance(known_face_encodings, unknown_face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # font = getFontSize(name,top, right, bottom, left)
        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 3, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

        #save in file
        if(name!="Unknown"):
            ts = time.time()      
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            Id=name.split('.')[0]
            only_name=name.split('.')[1]
            attendance.loc[len(attendance)] = [Id,only_name,date,timeStamp]
        else :
            print("unknown person(s)")
            count_unknown += 1
            res_mes="number of unknown faces detected : " + str(count_unknown)
            print(res_mes)
    
    
    # create attendance file
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="output/Punches/Punch_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    img_fileName="output/Punches/Punch_"+date+"_"+Hour+"-"+Minute+"-"+Second+".png"
    pil_image.save(img_fileName)
    attendance.to_csv(fileName,index=False)
    print("Punch: " + name);

    # Remove the drawing library from memory as per the Pillow docs
    del draw

    # Display the resulting image
    pil_image.show()



if __name__ == '__main__':
    args = sys.argv
    argc = len(args)

    needsHelp = False

    if (argc < 2):
    	needsHelp = True
    elif (args[1] == "-train"):
    	TrainImages()
    	quit()
    elif (args[1] == "-h" or args[1] == "-help" or args[1] == "-?"):
    	needsHelp = True
    elif (argc >= 3):
	    if (args[1] == "append"):
    		print("Not supported.")
    		quit()
	    elif (args[1] == "-check"):
    		TrackImages();

    print('Usage: %s\n' % args[0]
    	+ '\t-train: Train model for ALL associate faces.\n'
        + '\t-append <associate ID>: Adds an associates face data to the model.\n'
        + '\t-check <path-to-picture>: Checks an image for an associate.');
    quit()