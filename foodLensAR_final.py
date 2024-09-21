import sys
import cv2
import numpy as np
import os
import re
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont, QIcon
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
        self.image_label.setStyleSheet("border: 2px solid #2A2E32; padding: 5px;")
        self.camera_layout.addWidget(self.image_label)

        # Add a single button to start/stop the camera
        self.camera_button = QPushButton("Start Camera", self)
        self.camera_button.setIcon(QIcon('camera-icon.png'))  # Assuming you have an icon file
        self.camera_button.setStyleSheet(self.get_button_style())
        self.camera_button.clicked.connect(self.toggle_camera)
        self.camera_layout.addWidget(self.camera_button)

        # Add a button to start detection
        self.detect_button = QPushButton("Start Detection", self)
        self.detect_button.setIcon(QIcon('detect-icon.png'))
        self.detect_button.setStyleSheet(self.get_button_style())
        self.detect_button.clicked.connect(self.start_detection)
        self.camera_layout.addWidget(self.detect_button)

        # Add the camera layout to the main layout
        self.main_layout.addLayout(self.camera_layout)

        # QLabel to display detected items on the right side
        self.detected_items_label = QLabel(self)
        self.detected_items_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detected_items_label.setFont(QFont('Arial', 16))  # Set a bigger font size
        self.detected_items_label.setText("<b>Ingredient List:</b>")
        self.detected_items_label.setStyleSheet(self.get_ingredient_list_style())
        self.main_layout.addWidget(self.detected_items_label)

        # Timer to refresh the camera feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # Initialize camera and detection flag
        self.cap = None
        self.detecting = False
        self.camera_on = False  # Track camera state

        # Extract descriptors using SIFT
        self.kpListSIFT, self.desListSIFT = findSIFT(imgStock, sift)
        print('Number of descriptors (SIFT):', len(self.desListSIFT))

        # Detection count dictionary to store detection counts for each class
        self.detection_counts = {name: 0 for name in className}
        self.detection_threshold = 10  # Detection threshold

    def get_button_style(self):
        """Return a stylesheet for buttons."""
        return """
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """

    def get_detection_button_style(self, detecting):
        """Return a stylesheet for the detection button."""
        if detecting:
            return """
                QPushButton {
                    background-color: #f44336;
                    color: white;
                    font-size: 16px;
                    border-radius: 10px;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #e53935;
                }
            """
        else:
            return self.get_button_style()

    def get_ingredient_list_style(self):
        """Return a stylesheet for the ingredient list to make it prettier."""
        return """
            QLabel {
                border: 2px solid #2A2E32;
                padding: 15px;
                font-size: 14px;
                border-radius: 10px;
                background-color: #f9f9f9;
                margin-left: 10px;
            }
            QLabel::title {
                font-size: 18px;
                font-weight: bold;
                color: #333;
                padding-bottom: 10px;
            }
        """

    def toggle_camera(self):
        """Start or stop the camera feed."""
        if not self.camera_on:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        """Start the camera feed."""
        self.cap = cv2.VideoCapture(1)  # Use the correct device index for your camera 0:inbuilt 1:iruin
        self.timer.start(30)  # Refresh frame every 30 ms
        self.camera_on = True
        self.camera_button.setText("Stop Camera")
        self.camera_button.setIcon(QIcon('stop-icon.png'))  # Update icon for stopping
        self.camera_button.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                font-size: 16px;
                border-radius: 10px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #e53935;
            }
        """)

    def stop_camera(self):
        """Stop the camera feed."""
        self.timer.stop()
        if self.cap:
            self.cap.release()
        self.image_label.clear()
        self.camera_on = False
        self.camera_button.setText("Start Camera")
        self.camera_button.setIcon(QIcon('camera-icon.png'))  # Revert icon for starting
        self.camera_button.setStyleSheet(self.get_button_style())

    def start_detection(self):
        """Toggle detection on or off."""
        self.detecting = not self.detecting
        if self.detecting:
            self.detect_button.setText("Stop Detection")
        else:
            self.detect_button.setText("Start Detection")
            self.process_detection_results()

        # Update button color
        self.detect_button.setStyleSheet(self.get_detection_button_style(self.detecting))

    


    def process_detection_results(self):
        """Process detection results after detection stops."""
    
        # Create a set to track items that have already been added
        seen_items = set()

        # Filter out items below the threshold count, remove numbers from the end, and remove duplicates
        detected_items = []
        for item, count in self.detection_counts.items():
            if count >= self.detection_threshold:
                # Remove numbers from the end of the item
                cleaned_item = re.sub(r'\d+$', '', item).strip()

             # Only add the item if it's not already in the set (i.e., it's not a duplicate)
                if cleaned_item not in seen_items:
                    detected_items.append(cleaned_item)
                    seen_items.add(cleaned_item)

        # Create a pretty formatted ingredient list with bullet points
        detected_text = "<b>Ingredient List:</b> <br><ul>"
        for item in detected_items:
            detected_text += f"<li>{item}</li>"
        detected_text += "</ul>"

        # Print detected items in the console for debugging
        print("\nDetected Items (above threshold):")
        for item in detected_items:
            print(item)

        # Call the function to provide recipe suggestions based on detected items
        recipe_suggestion = self.get_recipe_suggestion(detected_items)

        # Show the recipe suggestion
        suggestion_text = f"<br><b>Recipe Suggestion:</b> <br>{recipe_suggestion}"
        self.detected_items_label.setText(detected_text + suggestion_text)

        # Reset detection counts after processing
        self.detection_counts = {name: 0 for name in self.detection_counts.keys()}

        
    def get_recipe_suggestion(self, detected_items):
        """Return and print recipe suggestions based on detected ingredients."""
        suggestions = []

       # Define known combinations and corresponding recipe suggestions
        if "pasta" in detected_items and "tomatopuree" in detected_items:
            suggestions.append("How about making some pasta with tomato sauce?")   
        if "chocolate" in detected_items:
            suggestions.append("Chocolate sounds like a nice snack!")
        if "bread" in detected_items and "eggs" in detected_items:
            suggestions.append("You could make an egg and bread recipe, like an egg Toast.")
        if "cocoa" in detected_items and "eggs" in detected_items and "chocolate" in detected_items:
            suggestions.append("It looks like you're halfway to a cake! Maybe buy some more ingredients")

        # Print and return all suggestions if any are found
        if suggestions:
            for suggestion in suggestions:
                print(suggestion)
            # Number the suggestions and format them for the GUI
            numbered_suggestions = "<br>".join([f"{idx + 1}. {suggestion}" for idx, suggestion in enumerate(suggestions)])
            return numbered_suggestions  # Return formatted and numbered suggestions for the GUI
    
        # Print and return the default message if no known combinations are found
        default_message = "Seems like you're missing some key ingredients. Why not buy a few more items?"
        print(default_message)
        return default_message

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

                    # Get points from the match
                    src_pts = np.float32([kp[0] for kp in kpPairs[i]]).reshape(-1, 1, 2)
                    dst_pts = np.float32([kp[1] for kp in kpPairs[i]]).reshape(-1, 1, 2)

                    # Compute homography
                    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

                    if M is not None:
                        h, w = imgStock[i].shape[:2]
                        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
                        dst = cv2.perspectiveTransform(pts, M)

                        # Calculate the bounding box size
                        x_min, y_min = np.min(dst[:, 0, :], axis=0)
                        x_max, y_max = np.max(dst[:, 0, :], axis=0)
                        bbox_width = x_max - x_min
                        bbox_height = y_max - y_min

                        if bbox_width > 20 and bbox_height > 20:
                            # Only consider this a valid detection if the bounding box is larger than 20x20 pixels
                            self.detection_counts[className[i]] += 1

                            # Draw bounding box and class name only if the box is large enough
                            img2 = cv2.polylines(img2, [np.int32(dst)], isClosed=True, color=(0, 255, 0), thickness=3)
                            top_left = tuple(np.int32(dst[0][0]))
                            cv2.putText(img2, className[i], top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                                        cv2.LINE_AA)

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
