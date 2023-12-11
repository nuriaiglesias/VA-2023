import matplotlib.pyplot as plt
import numpy as np
import cv2

image = cv2.imread("Material-Señales/00023.ppm")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

red_areas = cv2.bitwise_and(image, image, mask=mask_red)
blue_areas = cv2.bitwise_and(image, image, mask=mask_blue)

red_blurred = cv2.GaussianBlur(red_areas, (15, 15), 0)
blue_blurred = cv2.GaussianBlur(blue_areas, (15, 15), 0)

gray_red = cv2.cvtColor(red_areas, cv2.COLOR_BGR2GRAY)
gray_blue = cv2.cvtColor(blue_areas, cv2.COLOR_BGR2GRAY)

thresh_red = cv2.adaptiveThreshold(gray_red, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
thresh_blue = cv2.adaptiveThreshold(gray_blue, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

contours_red, _ = cv2.findContours(thresh_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(thresh_blue, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

detected_shapes = np.zeros_like(image)  

for cnt in contours_red:
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

    # Triángulo equilátero
    if len(approx) == 3:
        angles = []
        for i in range(3):
            pt1 = approx[i][0]
            pt2 = approx[(i + 1) % 3][0]
            pt3 = approx[(i + 2) % 3][0]
            vector1 = pt1 - pt2
            vector2 = pt3 - pt2
            dot = np.dot(vector1, vector2)
            det = np.linalg.norm(vector1) * np.linalg.norm(vector2)
            angle = np.arccos(dot / det) * (180 / np.pi)
            angles.append(angle)
        
        tolerance = 10 
        if all(abs(angle - 60) < tolerance for angle in angles):
            cv2.drawContours(detected_shapes, [cnt], 0, (0, 0, 255), 3)

# Definir el área total de la imagen
min_contour_area = 1000

for cnt in contours_blue:
    approx = cv2.approxPolyDP(cnt, 0.04 * cv2.arcLength(cnt, True), True)

    # Cuadrado
    if len(approx) == 4:
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w) / h

        contour_area = cv2.contourArea(cnt)
        rect_area = w * h

        if contour_area < rect_area and contour_area > min_contour_area and 0.7 <= aspect_ratio <= 1.3:
            cv2.drawContours(detected_shapes, [cnt], 0, (255, 0, 0), 3)

circles_red = cv2.HoughCircles(
    gray_red,
    cv2.HOUGH_GRADIENT,
    1,
    20,
    param1=100,
    param2=50,
    minRadius=1,
    maxRadius=40
)

if circles_red is not None:
    circles_red = np.uint16(np.around(circles_red))
    for circle in circles_red[0, :]:
        a, b, r = circle[0], circle[1], circle[2]
        cv2.circle(detected_shapes, (a, b), r, (0, 255, 0), 2)

circles_blue = cv2.HoughCircles(
    gray_blue,
    cv2.HOUGH_GRADIENT,
    1,
    20,
    param1=50,
    param2=30,
    minRadius=1,
    maxRadius=40
)

if circles_blue is not None:
    circles_blue = np.uint16(np.around(circles_blue))
    for circle in circles_blue[0, :]:
        a, b, r = circle[0], circle[1], circle[2]
        cv2.circle(detected_shapes, (a, b), r, (0, 255, 0), 2)


fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(cv2.cvtColor(red_areas, cv2.COLOR_BGR2RGB))
ax[0].set_title('Áreas Rojas detectadas')

ax[1].imshow(cv2.cvtColor(blue_areas, cv2.COLOR_BGR2RGB))
ax[1].set_title('Áreas Azules detectadas')

ax[2].imshow(cv2.cvtColor(detected_shapes, cv2.COLOR_BGR2RGB))
ax[2].set_title('Formas detectadas')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
