import cv2
import numpy as np
import os

# Specify path for stock images
pathStock = 'stockImages'
imgStock = []
className = []
myList = os.listdir(pathStock)
print('classes = ', len(myList))
bf = cv2.BFMatcher()

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
        desList.append((kp, des))
    return desList

# Image matching function
def findId(img, desList):
    kp2, des2 = orb.detectAndCompute(img, None)
    if des2 is None:
        return [], None, None, None
    
    bf = cv2.BFMatcher()
    matchList = []
    for kp, des in desList:
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
                goodMatch.append(m)
        matchList.append(len(goodMatch))
    
    if matchList:
        bestMatchIndex = np.argmax(matchList)
        return matchList, desList[bestMatchIndex][0], kp2, bestMatchIndex
    return matchList, None, None, None

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
    matchList, bestKp1, bestKp2, bestMatchIndex = findId(img2, desList)
    
    if matchList and bestKp1 is not None and bestKp2 is not None:
        if matchList[bestMatchIndex] > 0:
            print(f'Best match: {className[bestMatchIndex]} with {matchList[bestMatchIndex]} good matches')
            matches = bf.match(desList[bestMatchIndex][1], orb.detectAndCompute(img2, None)[1])
            imgOriginal = cv2.drawMatches(imgStock[bestMatchIndex], bestKp1, imgOriginal, bestKp2, matches[:20], None, flags=2)
            
            if bestKp2:
                points = np.array([kp.pt for kp in bestKp2], dtype=np.float32)
                if len(points) > 0:
                    x, y, w, h = cv2.boundingRect(points)
                    cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(imgOriginal, className[bestMatchIndex], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('img2', imgOriginal)
    
    # Handle key interrupts
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()

