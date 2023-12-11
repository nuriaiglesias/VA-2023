import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("Material-Señales/00446.ppm")

# Me falla el triangulo de la imagen 1 (me falta un trozo)
# Me falla el circulo del fondo de la imagen 159
# Me falla el circulo de arriba de la imagen 171

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

mask_red = cv2.inRange(hsv, lower_red, upper_red)
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

red_areas = cv2.bitwise_and(image, image, mask=mask_red)
blue_areas = cv2.bitwise_and(image, image, mask=mask_blue)


gray_red = cv2.cvtColor(red_areas, cv2.COLOR_BGR2GRAY)
gray_blue = cv2.cvtColor(blue_areas, cv2.COLOR_BGR2GRAY)

contours_red, _ = cv2.findContours(gray_red, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours_blue, _ = cv2.findContours(gray_blue, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

detected_shapes = np.zeros_like(image) 

def check_shape_red(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.3 * perimeter, True)
    num_sides = len(approx)
    area = cv2.contourArea(contour)
    if num_sides == 1 and area > 1300:
        return True
    elif num_sides == 1 and area > 250:
        return True
    elif num_sides == 2 and area > 50:
        return True
    elif num_sides >= 1 and num_sides <= 3 and area > 50:
        return True
    # elif num_sides == 2:  # Circulo 00171
    #     return True
    else:
        return False

def check_shape_blue(contour):
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)
    num_sides = len(approx)
    area = cv2.contourArea(contour)
    if num_sides == 4 and area > 1000: 
        return True
    elif num_sides == 6 and area > 750: 
        return True
    elif num_sides == 5 and area > 1900: 
        return True
    elif num_sides == 3 and area > 350:
        return True
    elif num_sides == 4 and area > 800:
        return True
    elif num_sides >= 2 and num_sides <= 4 and area > 100:
        return True
    elif num_sides == 5 and area > 650:
        return True
    elif num_sides == 5 and area > 200: 
        return True
    elif num_sides == 6 and area > 450:
        return True
    else:
        return False
    
for contour in contours_red:
    if check_shape_red(contour):
        cv2.drawContours(detected_shapes, [contour], 0, (0, 0, 255), 3)  

for contour in contours_blue:
    if check_shape_blue(contour):
        cv2.drawContours(detected_shapes, [contour], 0, (255, 0, 0), 3) 

image_contours_red = image.copy()
cv2.drawContours(image_contours_red, contours_red, -1, (0, 0, 255), 3)
image_contours_blue = image.copy()
cv2.drawContours(image_contours_blue, contours_blue, -1, (255, 0, 0), 3)

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(cv2.cvtColor(image_contours_red, cv2.COLOR_BGR2RGB))
ax[0].set_title('Áreas Rojas detectadas')

ax[1].imshow(cv2.cvtColor(image_contours_blue, cv2.COLOR_BGR2RGB))
ax[1].set_title('Áreas Azules detectadas')

ax[2].imshow(cv2.cvtColor(detected_shapes, cv2.COLOR_BGR2RGB))
ax[2].set_title('Formas detectadas')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()