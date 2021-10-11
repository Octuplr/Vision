import cv2
import sys
import os

###
#  Simple face detection and cropping tool.
###


class FaceCropper(object):
    CASCADE_PATH = "data/haarcascades/haarcascade_frontalface_default.xml"

    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(self.CASCADE_PATH)

    def generate(self, image_path, output_name, show_result = False):
        output_cords = False
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(img, 1.5, 6, minSize=(55, 55)) #Note: These parameters directly influence the accuracy and ability to detect a face.
        if (faces is None):
            print('Failed to detect face')
            return 0

        # Debugging, draw window showing detected faces.
        if (show_result):
            for (x, y, w, h) in faces:
                w=w*1.5
                h=h*1.5
                x1=int(x-(w/3))
                y1=int(y-(h/3))

                x2=int(x+(w))
                y2=int(y+(h))
                # x1=x
                # y1=y
                # x2=x+w
                # y2=y+h
                w=x2-x1
                h=y2-y1

                if (x1<0):
                    xOffset=x1*(-1)
                    x1=x1+xOffset+1
                    x2=x2-xOffset-1
                if (y1<0):
                    yOffset=y1*(-1)
                    y1=y1+yOffset+1
                    y2=y2-yOffset-1



                if (show_result):
                    print("X1: %s" % x1)
                    print("Y1: %s" % y1)
                    print("X2: %s" % x2)
                    print("Y2: %s" % y2)
                    print("W: %s" % w)
                    print("H: %s" % h)

                my=img.shape[0]
                mx=img.shape[1]
                print("Max X: %s" %mx)
                print("Max Y: %s" %my)

                cv2.rectangle(img, (x1,y1), (x2, y2), (255,0,0), 2)
                cv2.imshow('img', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        facecnt = len(faces)
        print("Detected faces: %d" % facecnt)
        i = 0
        height, width = img.shape[:2]

        for (x, y, w, h) in faces:
            w=w*1.5
            h=h*1.5
            x1=int(x-(w/3))
            y1=int(y-(h/3))

            x2=int(x+(w))
            y2=int(y+(h))
            
            w=x2-x1
            h=y2-y1



            r = max(w, h)/2
            centerx = (w/2)+x1
            centery = (h/2)+y1
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)

            if (nx<0):
                xOffset=nx*(-1)
                nx=nx+xOffset
                nr=nr-xOffset
                # nr=nr-xOffset
            if (ny<0):
                yOffset=y1*(-1)
                ny=ny+yOffset
                nr=nr-yOffset

            if (show_result):
                print("NX1: %s" % nx)
                print("NY1: %s" % ny)
                print("NX2: %s" % (nx+nr))
                print("NY2: %s" % (ny+nr))

            faceimg = img[ny:ny+nr, nx:nx+nr]
            lastimg = cv2.resize(faceimg, (256, 256))
            i += 1
            cv2.imwrite(output_name + "%d.jpg" % i, lastimg)
            
            if (output_cords):
                cords=open(output_name + "%d-cords.txt" % i, "w")
                cords.write(str(nx) + ":" + str(ny) + ":" + str((nx+nr)) + ":" + str((ny+nr)))
                cords.close()


        return facecnt


if __name__ == '__main__':
    args = sys.argv
    argc = len(args)

    if (argc != 2):
        print('Usage: %s [image file]' % args[0])
        quit()

    detecter = FaceCropper()
    detecter.generate(args[1], "output\\crop-face-data\\face")