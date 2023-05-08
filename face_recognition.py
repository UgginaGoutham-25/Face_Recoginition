 #Recognise Faces using some classification algorthim -KNN
#1. Read a Video stream using open cv
#2. exctract faces out of it(Testing)
#3.load the training data (nump arrays of all the persons)
  # x-values are stored in the numpy arrays
  # y-values we need to assign for each person
 #4. use knn to find the predication of face (int)
 #5. map the predicted id to  name of the user
 #6.Display the prediction on the screen - bounding box an name

import cv2
import numpy as np
import os

def distance(v1,v2):
	#Eucledian
	return np.sqrt(((v1-v2)**2).sum())



def knn(train, test, K=5):
	dist = []
 

	for i in range(train.shape[0]):
		# we need to get vector and label
		ix = train[i, :-1]
		iy = train[i, -1]
		#compute the distance from test point
		d = distance(test, ix)
		dist.append([d, iy])
	# sort based on distance and get top k
	dk = sorted(dist, key=lambda x: x[0])[:K]
	#Retrieve only labels
	labels = np.array(dk)[:, -1]

	# Get frequencies of each label
	output = np.unique(labels, return_counts=True)
	#Find max frequency and corresponding label
	index = np.argmax(output[1])
	return output[0][index]


#Intiilize the camera
cap = cv2.VideoCapture(0)

#Face Detection
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")#object that detects multiscaling  

skip = 0
dataset_path = './data/'

face_data = []
labels = []

class_id = 0#Labels for the Given file
names = {} #Mapping between id - name

#Data Prepartion
for fx in os.listdir(dataset_path):
	if fx.endswith('.npy'):
		#create a mapping between class_id  and name
		names[class_id] = fx[:-4]
		print("Loaded "+fx)
		data_item = np.load(dataset_path+fx)
		face_data.append(data_item)


		#Create Labels for the class
		target = class_id*np.ones((data_item.shape[0],))
		class_id += 1
		labels.append(target)

face_dataset = np.concatenate(face_data,axis=0)
face_labels = np.concatenate(labels,axis = 0).reshape((-1,1))


print(face_dataset.shape)
print(face_labels.shape)

trainset = np.concatenate((face_dataset,face_labels),axis=1)
print(trainset.shape)

#Testing

while True:
	ret,frame = cap.read()
	if ret == False:
		continue

	faces = face_cascade.detectMultiScale(frame,1.3,5)

	for face in faces:
		x,y,w,h = face

		#Get the face ROI
		offset = 10
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100))

		out = knn(trainset, face_section.flatten())

		#Display on the screen  and rectangle around it
		pred_name = names[int(out)]
		cv2.putText(frame,pred_name,(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

	cv2.imshow("Faces",frame)

	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()




