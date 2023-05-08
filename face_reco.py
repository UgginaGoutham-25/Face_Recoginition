import cv2
import numpy as np
import os

#Initialize the camera
cap = cv2.VideoCapture(0)

#Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml") #object that detects multiscaling  
       
skip = 0
face_data = []
dataset_path = './data/'

while True:
    ret, frame = cap.read()

    if ret == False:
        continue

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)
    faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
    
    if len(faces) > 0:
        for (x,y,w,h) in faces[-1]:
            cv2.rectangle(frame, (x,y), (x+w,y+h), (255,255,0), 2)

            #Extract (crop out the required face): region of Interest
            offset = 10
            face_section = frame[y-offset:y+h+offset, x-offset:x+w+offset]
            face_section = cv2.resize(face_section, (100,100))

            skip += 1
            if skip % 10 == 0:
                face_data.append(face_section)
                print(len(face_data))

        cv2.imshow("Frame", frame)
        key_pressed = cv2.waitKey(1) & 0xFF
        if key_pressed == ord('q'):
            break

#face_data = np.array(face_data)
#face_data = face_data.reshape((face_data.shape[0], -1))
#print(face_data.shape)

#np.save(dataset_path + 'face_data.npy', face_data)
#print("Data saved successfully")

cap.release()
cv2.destroyAllWindows()
