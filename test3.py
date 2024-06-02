import cv2
import numpy as np

# Function to detect keypoints and descriptors
def detect_features(image):
    # Initialize the ORB detector
    orb = cv2.ORB_create()

    # Find keypoints and descriptors
    keypoints, descriptors = orb.detectAndCompute(image, None)

    return keypoints, descriptors

# Function to match keypoints
def match_features(descriptors1, descriptors2):
    # Initialize the matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Match descriptors
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    return matches

# Function to draw bounding boxes around matched keypoints
def draw_matches(img1, img2, keypoints1, keypoints2, matches):
    # Draw matches
    img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return img_matches

# Load the image of the object to be recognized
object_image = cv2.imread('kcfiles/img/egg2.jpg', cv2.IMREAD_GRAYSCALE)

# Detect keypoints and descriptors of the object image
object_keypoints, object_descriptors = detect_features(object_image)

# Initialize video capture
cap = cv2.VideoCapture('kcfiles/vid/vid1.mp4') # Replace 'kcfiles/vid/vid1.mp4' with your video file

# Get the dimensions of the video frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the desired screen width and height
desired_width = 800
desired_height = int(desired_width * frame_height / frame_width)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect keypoints and descriptors of the frame
    frame_keypoints, frame_descriptors = detect_features(frame_gray)

    # Match features between the object image and the frame
    matches = match_features(object_descriptors, frame_descriptors)

    # Draw matches and bounding box around the object
    img_matches = draw_matches(object_image, frame_gray, object_keypoints, frame_keypoints, matches)

    # Resize the frame to desired dimensions
    resized_frame = cv2.resize(img_matches, (desired_width, desired_height))

    # Display the resulting frame
    cv2.imshow('Object Recognition', resized_frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
