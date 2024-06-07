import cv2
import numpy as np
import os

# Specify path for stock images
pathStock = 'stockImages'
imgStock = []
className = []
myList = os.listdir(pathStock)
print('classes = ', len(myList))

# Default features = 500, we choose 1000
orb = cv2.ORB_create(nfeatures=1000)

# Load stock images and class names
for cl in myList:
    imgCurrent = cv2.imread(f'{pathStock}/{cl}', 0)
    imgStock.append(imgCurrent)
    className.append(os.path.splitext(cl)[0])
    print(cl)

# Descriptor extraction function
def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

# Image matching function
def findId(img, desList):
    kp2, des2 = orb.detectAndCompute(img, None)
    if des2 is None:
        return []
    
    bf = cv2.BFMatcher()
    matchList = []
    for des in desList:
        if des is None:
            matchList.append(0)
            continue
        
        # Convert descriptors to float32 if needed
        if des.dtype != 'float32':
            des = des.astype('float32')
        if des2.dtype != 'float32':
            des2 = des2.astype('float32')
        
        matches = bf.knnMatch(des, des2, k=2)  # k=2
        goodMatch = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                goodMatch.append([m])
        matchList.append(len(goodMatch))
    return matchList

# Extract descriptors from stock images
desList = findDes(imgStock)
print('Number of descriptors:', len(desList))

cap = cv2.VideoCapture(1)

while True:
    success, img2 = cap.read()
    if not success:
        break
    
    imgOriginal = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    matchList = findId(img2, desList)
    
    if matchList:
        bestMatchIndex = np.argmax(matchList)
        print(f'Best match: {className[bestMatchIndex]} with {matchList[bestMatchIndex]} good matches')
    
    cv2.imshow('img2', imgOriginal)
    
    # Handle key interrupts
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
