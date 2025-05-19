import cv2
import numpy as np

# Load the image
img = cv2.imread('brain3.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Threshold the image
_, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

# Find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours based on area
tumor_contours = []
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 500: # Adjust threshold as needed
       tumor_contours.append(contour)

# Draw contours on the original image 
cv2.drawContours(img, tumor_contours, -1, (0, 255, 0), 2)

# Display the image
cv2_imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
