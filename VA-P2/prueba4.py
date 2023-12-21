import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("Material Señales/00001.ppm")

# Me falla el circulo de la imagen 115

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 50, 50])
upper_red = np.array([20, 255, 255])

lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

red_areas = cv2.bitwise_and(image, image, mask=mask_red)
blue_areas = cv2.bitwise_and(image, image, mask=mask_blue)

gray_red = cv2.cvtColor(red_areas, cv2.COLOR_BGR2GRAY)
ret,binary_red = cv2.threshold(gray_red,120,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
gray_blue = cv2.cvtColor(blue_areas, cv2.COLOR_BGR2GRAY)
ret,binary_blue = cv2.threshold(gray_blue,120,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

min_area_threshold_red = 1

def connected_components_red(morph_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph_image, connectivity=8)
    connected_image = np.zeros_like(morph_image)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area_threshold_red:
            connected_image[labels == i] = 255
    return connected_image

min_area_threshold_blue = 1

def connected_components_blue(morph_image):
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(morph_image, connectivity=8)
    connected_image = np.zeros_like(morph_image)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] > min_area_threshold_blue:
            connected_image[labels == i] = 255
    return connected_image

connected_red = connected_components_red(binary_red)
connected_blue = connected_components_blue(binary_blue)

contours_red, _ = cv2.findContours(connected_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(connected_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

image_contours_red = image.copy()
image_contours_blue = image.copy()

circles_red = cv2.HoughCircles(connected_red, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=100, param2=10, minRadius=10, maxRadius=36)
# 01
circles_blue = cv2.HoughCircles(connected_blue, cv2.HOUGH_GRADIENT, dp=1, minDist=30, param1=80, param2=10, minRadius=4, maxRadius=40) 

# 23
# 40
# 115
# 159
# 171
# 177
# 219
# 235
# 262
# 446

# if circles_red is not None:
#     circles_red = np.uint16(np.around(circles_red))
#     for circle in circles_red[0, :]:
#         center = (circle[0], circle[1])
#         radius = circle[2]
#         cv2.circle(image_contours_red, center, radius, (0, 255, 0), 2)

# if circles_blue is not None:
#     circles_blue = np.uint16(np.around(circles_blue))
#     for circle in circles_blue[0, :]:
#         center = (circle[0], circle[1])
#         radius = circle[2]
#         cv2.circle(image_contours_blue, center, radius, (0, 255, 0), 2)

# if circles_blue is not None:
#     circles_blue = np.uint16(np.around(circles_blue))

#     for circle in circles_blue[0, :]:
#         center = (circle[0], circle[1])
#         radius = circle[2]

#         x, y = center[0], center[1]

#         # Calcular límites del área del círculo
#         top_y = max(0, y - radius)
#         bottom_y = min(image.shape[0], y + radius)
#         left_x = max(0, x - radius)
#         right_x = min(image.shape[1], x + radius)

#         # Verificar que los límites estén dentro de la imagen
#         if top_y < bottom_y and left_x < right_x:
#             circle_area = image[top_y:bottom_y, left_x:right_x]

#             if circle_area.size > 0:  # Verificar si el área recortada no está vacía
#                 hsv_circle_area = cv2.cvtColor(circle_area, cv2.COLOR_BGR2HSV)
#                 mask_blue_circle_area = cv2.inRange(hsv_circle_area, lower_blue, upper_blue)
#                 mask_white_circle_area = cv2.inRange(hsv_circle_area, lower_white, upper_white)  # Definir los límites de blanco

#                 blue_pixels_percentage = np.sum(mask_blue_circle_area == 255) / (circle_area.shape[0] * circle_area.shape[1]) * 100
#                 white_pixels_percentage = np.sum(mask_white_circle_area == 255) / (circle_area.shape[0] * circle_area.shape[1]) * 100

#                 # Verificar si más del 45% del área es azul y más del 45% es blanco
#                 if blue_pixels_percentage > 45 and white_pixels_percentage > 5:
#                     cv2.circle(image_contours_blue, center, radius, (0, 255, 0), 2)

if circles_blue is not None:
    circles_blue = np.uint16(np.around(circles_blue))

    for circle in circles_blue[0, :]:
        center = (circle[0], circle[1])
        radius = circle[2]

        x, y = center[0], center[1]

        top_y = max(0, y - radius)
        bottom_y = min(image.shape[0], y + radius)
        left_x = max(0, x - radius)
        right_x = min(image.shape[1], x + radius)

        if top_y < bottom_y and left_x < right_x:
            circle_area = image[top_y:bottom_y, left_x:right_x]

            if circle_area.size > 0:  
                hsv_circle_area = cv2.cvtColor(circle_area, cv2.COLOR_BGR2HSV)
                mask_blue_circle_area = cv2.inRange(hsv_circle_area, lower_blue, upper_blue)

                mask_blue_circle_area = cv2.inRange(hsv_circle_area, lower_blue, upper_blue)
                mask_white_circle_area = cv2.inRange(hsv_circle_area, lower_white, upper_white) 

                blue_pixels_percentage = np.sum(mask_blue_circle_area == 255) / (circle_area.shape[0] * circle_area.shape[1]) * 100
                white_pixels_percentage = np.sum(mask_white_circle_area == 255) / (circle_area.shape[0] * circle_area.shape[1]) * 100

                if blue_pixels_percentage > 5:
                    _, thresh = cv2.threshold(mask_blue_circle_area, 127, 255, cv2.THRESH_BINARY)
                    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    for contour in contours:
                        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)
                        
                        if len(approx) <= 10 and len(approx) >= 1: 
                            area = cv2.contourArea(contour)
                            if area > 1:
                                cv2.circle(image_contours_blue, center, radius, (0, 255, 0), 2) 

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(cv2.cvtColor(image_contours_red, cv2.COLOR_BGR2RGB))
ax[0].set_title('Áreas Rojas detectadas')

ax[1].imshow(cv2.cvtColor(image_contours_blue, cv2.COLOR_BGR2RGB))
ax[1].set_title('Áreas Azules detectadas')

ax[2].imshow(connected_blue)
ax[2].set_title('Áreas Rojas detectadas')


for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()