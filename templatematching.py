import cv2
import numpy as np

# Video laden
video_path = 'input/eggs.mp4'
cap = cv2.VideoCapture(video_path)

# Liste der Vorlagenbilder
template_path = "templateData/uncropped/"
template_images = [template_path + 'eggs1.jpeg', template_path + 'eggs2.jpeg', template_path + 'eggs3.jpeg']

# Funktion zur Erstellung einer Bildpyramide
def create_image_pyramid(image, max_level):
    pyramid = [image]
    for i in range(1, max_level):
        image = cv2.pyrDown(image)
        pyramid.append(image)
    return pyramid

# Lade und konvertiere die Vorlagenbilder in Graustufen und erstelle Pyramiden
templates = [cv2.imread(template, cv2.IMREAD_GRAYSCALE) for template in template_images]
template_pyramids = [create_image_pyramid(template, 3) for template in templates]

# Funktion, um die beste Übereinstimmung zu finden
def find_best_match(frame, template_pyramids):
    best_match = None
    best_val = -1
    best_loc = None
    best_template_size = None

    for template_pyramid in template_pyramids:
        for level, template in enumerate(template_pyramid):
            # Überprüfen, ob das Template kleiner oder gleich groß wie der Frame ist
            if template.shape[0] > frame.shape[0] or template.shape[1] > frame.shape[1]:
                continue

            res = cv2.matchTemplate(frame, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            if max_val > best_val:
                best_val = max_val
                best_loc = max_loc
                best_match = template
                best_template_size = (template.shape[1], template.shape[0])
                best_level = level

    return best_match, best_loc, best_template_size, best_level

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konvertiere den Frame in Graustufen und erstelle eine Pyramide
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_pyramid = create_image_pyramid(gray_frame, 3)

    # Finde die beste Übereinstimmung auf verschiedenen Ebenen der Pyramide
    best_match, best_loc, best_template_size, best_level = None, None, None, None
    for level, pyramid_frame in enumerate(frame_pyramid):
        match, loc, size, level = find_best_match(pyramid_frame, template_pyramids)
        if match is not None:
            best_match, best_loc, best_template_size, best_level = match, loc, size, level
            break

    if best_match is not None:
        w, h = best_template_size
        # Skaliere die Koordinaten zurück auf die ursprüngliche Bildgröße
        scale_factor = 2 ** best_level
        top_left = (int(best_loc[0] * scale_factor), int(best_loc[1] * scale_factor))
        bottom_right = (top_left[0] + int(w * scale_factor), top_left[1] + int(h * scale_factor))

        # Zeichne ein Rechteck um die gefundene Position
        cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

    # Zeige den Frame mit der Markierung
    cv2.imshow('Template Matching', frame)

    # Beenden, wenn die 'q'-Taste gedrückt wird
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
