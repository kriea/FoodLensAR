import cv2
import numpy as np
import os

#lets specify path for stock images
pathStock ='stockImages'
imgStock = []
className = []
myList = os.listdir(pathStock)
print('classes = ', len(myList))

#default features = 500, we choose 1000
orb = cv2.ORB_create(nfeatures=1000)

 #classes
for cl in myList:
    imgCurrent = cv2.imread(f'{pathStock}/{cl}',0)
    imgStock.append(imgCurrent)
    className.append(os.path.splitext(cl)[0])
    print(cl)
    
#descriptors
def findDes(img):
    desList=[]
    for img in img:
        kp,des = orb.detectAndCompute(img,None)
        desList.append(des)
    return desList

def findId(img,desList):
    kp2,des2 = orb.detectAndCompute(img,None)
    bf = cv2.BFMatcher()
    matchList = []
    for des in desList:
        matches = bf.knnMatch(des,des2,k=2) ##k=2
        goodMatch = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            goodMatch.append([m])
    matchList.append(len(goodMatch))
    print(matchList)
    
    
    
    


desList = findDes(imgStock)
print(len(desList))

cap = cv2.VideoCapture(0)

while True:
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    findId(img2,desList)
    cv2.imshow('img2',imgOriginal)
    
    
    cv2.waitKey(1)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    
# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
