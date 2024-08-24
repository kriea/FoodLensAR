import sys
import cv2
import numpy as np
import os
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton

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
    return matchList, kp2, kpPairs


class CameraWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("FoodLens AR™")
        self.setGeometry(100, 100, 1000, 600)  # Adjusted window size

        # Main widget and layout
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)
        self.main_layout = QHBoxLayout(self.main_widget)  # Use horizontal layout

        # Layout for camera input and buttons
        self.camera_layout = QVBoxLayout()

        # QLabel to display the camera feed
        self.image_label = QLabel(self)
        self.camera_layout.addWidget(self.image_label)

        # Add a button to start the camera
        self.start_button = QPushButton("Start Camera", self)
        self.start_button.clicked.connect(self.start_camera)
        self.camera_layout.addWidget(self.start_button)

        # Add a button to stop the camera
        self.stop_button = QPushButton("Stop Camera", self)
        self.stop_button.clicked.connect(self.stop_camera)
        self.camera_layout.addWidget(self.stop_button)

        # Add a button to start detection
        self.detect_button = QPushButton("Start Detection", self)
        self.detect_button.clicked.connect(self.start_detection)
        self.camera_layout.addWidget(self.detect_button)

        # Add the camera layout to the main layout
        self.main_layout.addLayout(self.camera_layout)

        # QLabel to display detected items on the right side
        self.detected_items_label = QLabel(self)
        self.detected_items_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detected_items_label.setFont(QFont('Arial', 16))  # Set a bigger font size
        self.detected_items_label.setText("Ingredient List:")
        self.main_layout.addWidget(self.detected_items_label)

        # Timer to refresh the camera feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Initialize camera and detection flag
        self.cap = None
        self.detecting = False

        # Extract descriptors using SIFT
        self.kpListSIFT, self.desListSIFT = findSIFT(imgStock, sift)
        print('Number of descriptors (SIFT):', len(self.desListSIFT))

        # Detection count dictionary to store detection counts for each class
        self.detection_counts = {name: 0 for name in className}
        self.detection_threshold = 10  # Detection threshold

    def start_camera(self):
        """Start the camera feed."""
        self.cap = cv2.VideoCapture(0)  # Use the correct device index for your camera
        self.timer.start(30)  # Refresh frame every 30 ms

    def stop_camera(self):
        """Stop the camera feed."""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.image_label.clear()

    def start_detection(self):
        """Toggle detection on or off."""
        self.detecting = not self.detecting
        if self.detecting:
            self.detect_button.setText("Stop Detection")
        else:
            self.detect_button.setText("Start Detection")
            self.process_detection_results()

    def process_detection_results(self):
        """Process detection results after detection stops."""
        # Filter out items below the threshold count
        detected_items = [item for item, count in self.detection_counts.items() if count >= self.detection_threshold]

        # Update the label with detected items
        detected_text = "Ingredient List: \n" + "\n".join(detected_items)
        self.detected_items_label.setText(detected_text)

        print("\nDetected Items (above threshold):")
        for item in detected_items:
            print(item)

        # Reset detection counts after processing
        self.detection_counts = {name: 0 for name in className}

    def update_frame(self):
        """Capture frame from camera and update QLabel."""
        if not self.cap:
            return

        success, img2 = self.cap.read()
        if not success:
            return

        imgOriginal = img2.copy()
        img2Gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Use SIFT to find matches
        if self.detecting:
            matchList, kp2, kpPairs = findSIFTId(img2Gray, self.kpListSIFT, self.desListSIFT, sift)

            for i, matchCount in enumerate(matchList):
                if matchCount > 60:  # Threshold for good matches
                    print(f'Match found (SIFT): {className[i]} with {matchCount} good matches')

                    # Update detection count for the detected item
                    self.detection_counts[className[i]] += 1

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

        # Convert the image from BGR to RGB format
        rgb_image = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w

        # Convert the image to QImage format
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Set the QImage to the QLabel
        self.image_label.setPixmap(QPixmap.fromImage(qt_image))


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Create and show the main window
    window = CameraWindow()
    window.show()

    # Execute the application
    sys.exit(app.exec_())
