import cv2
import numpy as np
from imutils import contours
import os

def extract_letter_from_plate(save_path, filepath, imshow=False):
    # Read the image
    image = cv2.imread(filepath)
    if image is None:
        print(f"Error: Unable to read image at {filepath}")
        return

    # Prepare image
    mask = np.zeros(image.shape, dtype=np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours
    cnts, hierarchy = cv2.findContours(morph_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    image_copy = image.copy()
    cv2.drawContours(image_copy, cnts, -1, (255,1,1), 2)
    if imshow:
        cv2.imshow("Image", image_copy)

    # Sort contours left-to-right
    (cnts, _) = contours.sort_contours(cnts, method="left-to-right")

    # Prepare save directory
    filename = os.path.basename(filepath)
    folder_name = os.path.splitext(filename)[0]
    new_dir = os.path.join(save_path, folder_name)
    os.makedirs(new_dir, exist_ok=True)

    # Extract and save ROI
    ROI_number = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if 100 < area < 900:
            x, y, w, h = cv2.boundingRect(c)
            ratio = h / float(w)
            if 0.89 < ratio < 5:
                ROI = thresh[y:y+h, x:x+w]
                cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)
                save_path_roi = os.path.join(new_dir, f'ROI_{ROI_number}.jpg')
                cv2.imwrite(save_path_roi, ROI)
                ROI_number += 1

    if imshow:
        cv2.imshow('Mask', mask)
        cv2.imshow('Morph', morph_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Successfully extracted {ROI_number} letters from plate and saved to: {new_dir}")

