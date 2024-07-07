import cv2
import numpy as np
import os

# Specify path for stock images
pathStock = 'stockImages/daves'
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
    kpList = []
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        desList.append(des)
        kpList.append(kp)
    return kpList, desList


# Function to perform SIFT matching
def findSIFTId(img, kpList, desList, sift):
    kp2, des2 = sift.detectAndCompute(img, None)
    if des2 is None or len(des2) == 0:
        return [], [], []

    flann = cv2.FlannBasedMatcher()
    matchList = []
    kpPairs = []
    for kp, des in zip(kpList, desList):
        if des is None or len(des) == 0:
            matchList.append(0)
            kpPairs.append([])
            continue

        # Convert descriptors to float32 if needed
        if des.dtype != 'float32':
            des = des.astype('float32')
        if des2.dtype != 'float32':
            des2 = des2.astype('float32')

        matches = flann.knnMatch(des, des2, k=2)
        goodMatch = []
        goodKpPairs = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                goodMatch.append(m)
                goodKpPairs.append((kp[m.queryIdx].pt, kp2[m.trainIdx].pt))
        matchList.append(len(goodMatch))
        kpPairs.append(goodKpPairs)
    return matchList, kp2, kpPairs#


# # Debug: Draw matches
# matches_img = cv2.drawMatches(imgStock[i], kpListSIFT[i], img2, kp2, allGoodMatches[i], None,
#                               matchesMask=matchesMask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# cv2.imshow('Matches', matches_img)


# Extract descriptors using SIFT
kpListSIFT, desListSIFT = findSIFT(imgStock, sift)
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
    matchList, kp2, kpPairs = findSIFTId(img2Gray, kpListSIFT, desListSIFT, sift)

    for i, matchCount in enumerate(matchList):
        if matchCount > 60:  # Threshold for good matches
            print(f'Match found (SIFT): {className[i]} with {matchCount} good matches')

            # Get points from the match
            src_pts = np.float32([kp[0] for kp in kpPairs[i]]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp[1] for kp in kpPairs[i]]).reshape(-1, 1, 2)

            # Compute homography
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

            if M is not None:
                h, w = imgStock[i].shape[:2]
                pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                dst = cv2.perspectiveTransform(pts, M)

                # Draw bounding box
                img2 = cv2.polylines(img2, [np.int32(dst)], isClosed=True, color=(0, 255, 0), thickness=3)
                # Draw the class name
                top_left = tuple(np.int32(dst[0][0]))
                cv2.putText(img2, className[i], top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


    # Display video frame
    cv2.imshow('img2', img2)

    # Handle key interrupts
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
