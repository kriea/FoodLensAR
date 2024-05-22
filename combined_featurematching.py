import cv2
import numpy as np

# Video laden
video_path = 'input/input.mp4'
cap = cv2.VideoCapture(video_path)

# Liste der Vorlagenbilder
egg_template_path = "templateData/"
banana_template_path = "templateData/"
egg_template_images = [egg_template_path + 'eggs1.jpeg', egg_template_path + 'eggs2.jpeg', egg_template_path + 'eggs3.jpeg',
                       egg_template_path + 'eggs4.jpeg', egg_template_path + 'eggs5.jpeg', egg_template_path + 'eggs6.jpeg',
                       egg_template_path + 'eggs7.jpeg', egg_template_path + 'eggs8.jpeg']
banana_template_images = [banana_template_path + 'milk1.jpeg']

# ORB-Detektor initialisieren
orb = cv2.ORB_create()

# Lade und konvertiere die Vorlagenbilder in Graustufen und finde ORB-Merkmale
egg_templates = [cv2.imread(template, cv2.IMREAD_GRAYSCALE) for template in egg_template_images]
banana_templates = [cv2.imread(template, cv2.IMREAD_GRAYSCALE) for template in banana_template_images]
egg_keypoints_and_descriptors = [orb.detectAndCompute(template, None) for template in egg_templates]
banana_keypoints_and_descriptors = [orb.detectAndCompute(template, None) for template in banana_templates]

# BFMatcher initialisieren
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Parameters for smoothing
SMOOTHING_WINDOW_SIZE = 15
egg_positions = []
banana_positions = []

# Minimale und maximale Größe der Bounding Box festlegen
MIN_BOX_WIDTH = 50  # Verkleinern Sie die Mindestgröße
MIN_BOX_HEIGHT = 50  # Verkleinern Sie die Mindestgröße
MAX_BOX_WIDTH = 300  # Verkleinern Sie die Maximalgröße
MAX_BOX_HEIGHT = 300  # Verkleinern Sie die Maximalgröße

def match_features(frame, keypoints_and_descriptors, ratio=0.75):
    frame_keypoints, frame_descriptors = orb.detectAndCompute(frame, None)
    all_points = []

    for template_kp, template_desc in keypoints_and_descriptors:
        matches = bf.match(template_desc, frame_descriptors)
        matches = sorted(matches, key=lambda x: x.distance)
        # Nur die besten Matches auswählen (hier die besten 10%)
        num_good_matches = int(len(matches) * ratio)
        matches = matches[:num_good_matches]
        points = np.float32([frame_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)

        all_points.extend(points)

    return np.array(all_points)

def draw_dynamic_bounding_box(frame, points, positions, min_box_width, min_box_height, max_box_width, max_box_height, color):
    if len(points) == 0:
        return frame

    # Berechne den Mittelpunkt der Punkte
    center = np.mean(points, axis=0).astype(int)

    # Füge die aktuelle Position zur Liste der Positionen hinzu
    positions.append(center)
    if len(positions) > SMOOTHING_WINDOW_SIZE:
        positions.pop(0)

    # Berechne den gleitenden Durchschnitt der letzten Positionen
    smoothed_center = np.mean(positions, axis=0).astype(int)

    # Berechne die Bounding Box Größe basierend auf der Varianz der Punkte
    if len(points) > 1:
        x_coords, y_coords = points[:, 0], points[:, 1]
        x_min, x_max = np.min(x_coords), np.max(x_coords)
        y_min, y_max = np.min(y_coords), np.max(y_coords)
        box_width = int(x_max - x_min)
        box_height = int(y_max - y_min)

        # Begrenze die Größe der Bounding Box auf die minimalen und maximalen Werte
        box_width = max(min(box_width, max_box_width), min_box_width)
        box_height = max(min(box_height, max_box_height), min_box_height)
    else:
        box_width = min_box_width
        box_height = min_box_height

    x = int(smoothed_center[0] - box_width / 2)
    y = int(smoothed_center[1] - box_height / 2)

    # Zeichne die dynamische Bounding Box
    cv2.rectangle(frame, (x, y), (x + box_width, y + box_height), color, 2)
    return frame

cv2.namedWindow('Feature Matching', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konvertiere den Frame in Graustufen
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Finde die beste Übereinstimmung der Merkmale für Eier
    egg_points = match_features(gray_frame, egg_keypoints_and_descriptors, ratio=0.1)
    # Finde die beste Übereinstimmung der Merkmale für die Milchverpackung
    banana_points = match_features(gray_frame, banana_keypoints_and_descriptors, ratio=0.1)

    if len(egg_points) > 0:
        # Zeichne die dynamische Bounding Box basierend auf den Matches für Eier
        frame_with_egg_box = draw_dynamic_bounding_box(frame, egg_points, egg_positions, MIN_BOX_WIDTH, MIN_BOX_HEIGHT,
                                                       MAX_BOX_WIDTH, MAX_BOX_HEIGHT, (0, 255, 0))
    else:
        frame_with_egg_box = frame

    if len(banana_points) > 0:
        # Zeichne die dynamische Bounding Box basierend auf den Matches für die Milchverpackung
        frame_with_banana_box = draw_dynamic_bounding_box(frame_with_egg_box, banana_points, banana_positions,
                                                          MIN_BOX_WIDTH, MIN_BOX_HEIGHT, MAX_BOX_WIDTH, MAX_BOX_HEIGHT, (255, 0, 0))
    else:
        frame_with_banana_box = frame_with_egg_box

    # Zeige den Frame mit den Markierungen und Bounding Boxen
    cv2.imshow('Feature Matching', frame_with_banana_box)

    # Beenden, wenn die 'q'-Taste gedrückt wird oder das Fenster geschlossen wird
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Feature Matching', cv2.WND_PROP_VISIBLE) < 1:
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
