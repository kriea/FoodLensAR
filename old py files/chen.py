import cv2
import numpy as np
import os

# Specify path for stock images
pathStock = 'stockImages'
imgStock = []
className = []
myList = os.listdir(pathStock)
print('classes = ', len(myList))

# Create SIFT detector
sift = cv2.SIFT_create(nfeatures=1000)

# Load stock images and class names
for cl in myList:
    imgCurrent = cv2.imread(f'{pathStock}/{cl}', 0)
    imgStock.append(imgCurrent)
    className.append(os.path.splitext(cl)[0])
    print(cl)

# Function to extract SIFT descriptors
def findSIFT(images, sift):
    desList = []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        desList.append(des)
    return desList

# Function to perform SIFT matching
def findSIFTId(img, desList, sift):
    kp2, des2 = sift.detectAndCompute(img, None)
    if des2 is None or len(des2) == 0:
        return []

    flann = cv2.FlannBasedMatcher()
    matchList = []
    for des in desList:
        if des is None or len(des) == 0:
            matchList.append(0)
            continue
        
        # Convert descriptors to float32 if needed
        if des.dtype != 'float32':
            des = des.astype('float32')
        if des2.dtype != 'float32':
            des2 = des2.astype('float32')
        
        matches = flann.knnMatch(des, des2, k=2)
        goodMatch = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                goodMatch.append([m])
        matchList.append(len(goodMatch))
    return matchList


# Extract descriptors using SIFT
desListSIFT = findSIFT(imgStock, sift)
print('Number of descriptors (SIFT):', len(desListSIFT))

# Initialize video capture
cap = cv2.VideoCapture(0)

while True:
    success, img2 = cap.read()
    if not success:
        break
    
    imgOriginal = img2.copy()
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    
    
    # Use SIFT to find matches
    matchList = findSIFTId(img2Gray, desListSIFT, sift)
    
    if matchList:
        bestMatchIndex = np.argmax(matchList)
        if matchList[bestMatchIndex] > 20:
            print(f'Best match (SIFT): {className[bestMatchIndex]} with {matchList[bestMatchIndex]} good matches')
            
            # Perform perspective transformation
            src_pts = np.float32([[0, 0], [0, imgStock[bestMatchIndex].shape[0] - 1],
                                  [imgStock[bestMatchIndex].shape[1] - 1, imgStock[bestMatchIndex].shape[0] - 1],
                                  [imgStock[bestMatchIndex].shape[1] - 1, 0]]).reshape(-1, 1, 2)
            dst_pts = np.float32([[0, 0], [0, img2.shape[0] - 1], [img2.shape[1] - 1, img2.shape[0] - 1],
                                  [img2.shape[1] - 1, 0]]).reshape(-1, 1, 2)
            
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = imgStock[bestMatchIndex].shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            
            img2 = cv2.polylines(img2, [np.int32(dst)], True, (255, 0, 0), 3, cv2.LINE_AA)
    
    # Display video frame
    cv2.imshow('img2', imgOriginal)
    
    # Handle key interrupts
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()


