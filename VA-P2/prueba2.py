import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

# Cargar la imagen
image = cv2.imread("Material Señales/00446.ppm")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Me detecta circulos azules en cuadrados azules en la 235 e igual en la 262

lower_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

# Crear una máscara para las zonas azules
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Aplicar un desenfoque para reducir el ruido en la máscara
blurred_mask = cv2.GaussianBlur(mask, (5, 5), 0)

# Detección de círculos usando HoughCircles
circles = cv2.HoughCircles(
    blurred_mask,
    cv2.HOUGH_GRADIENT_ALT,
    dp=1,
    minDist=10,
    param1=100,
    param2=0.5,
    minRadius=1,
    maxRadius=50
)

canvas = np.zeros_like(image)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        x, y, r = circle
        # Asegurarse de que los valores no estén fuera de los límites de la imagen
        x = np.clip(x, r, gray.shape[1] - r)
        y = np.clip(y, r, gray.shape[0] - r)

        # Recortar la región del círculo para buscar triángulos dentro
        roi = gray[max(0, y - r):min(gray.shape[0], y + r), max(0, x - r):min(gray.shape[1], x + r)]
        
        roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Crear máscaras para el blanco y azul
        mask_blue = cv2.inRange(roi_hsv, lower_blue, upper_blue)
        mask_white = cv2.inRange(roi_hsv, lower_white, upper_white)
        combined_mask = cv2.bitwise_or(mask_blue, mask_white)

        masked_roi = cv2.bitwise_and(roi, roi, mask=combined_mask)
        # Dibujar el círculo y mostrar la región segmentada
        # if cv2.countNonZero(combined_mask) > 105: 446
        # if cv2.countNonZero(combined_mask) > 80: # 001 - 115
        # if cv2.countNonZero(combined_mask) > 1: # 159 - lo quito
        edges = cv2.Canny(mask_white, 50, 150, apertureSize=3)
        if cv2.countNonZero(edges) > 0:
                if edges.size > 0:
                    cv2.imshow("edges", edges)
                    cv2.waitKey(0) 
                    cv2.destroyAllWindows()
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        # if area > 110: # 446 
                        # if area > 1: 001
                        # if area > 1: 159 - lo quito
                        peri = cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                        # if len(approx) >= 6 and len(approx) <= 6: # 446
                        # if len(approx) >= 8 and len(approx) <= 10: 001 - 115
                        if len(approx) >= 2 and len(approx) <= 10:
                            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
                            cv2.circle(canvas, (x, y), r, (0, 255, 0), 2)

cv2.imshow("Detected Arrows", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
 

# Mostrar la imagen con los círculos y líneas detectadas
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(blurred_mask)
ax[0].set_title('Áreas Rojas detectadas')

ax[2].imshow(image)
ax[2].set_title('Áreas Rojas detectadas')


for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
