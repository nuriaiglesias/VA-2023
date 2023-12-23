import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
# Cargar la imagen
image = cv2.imread("Material Señales/00446.ppm")
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# circulos azules:
#       numero de fallos: 4 + 1 + 1 + 1 + 2 + 1 + 2 + 8

# Definir límites para tonos de azul más amplios
lower_blue = np.array([106, 50, 50])  # Valores inferiores para tonos azules más oscuros
upper_blue = np.array([130, 165, 255])  # Valores superiores para tonos azules más claros

lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 30, 255])

# Crear una máscara solo para los tonos azules ampliados
mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

# Aplicar un aumento en la intensidad de los tonos azules
intensity_factor = 1.9  # Factor de aumento de intensidad (puedes ajustar este valor)
enhanced_blue = np.where(mask_blue > 0, np.clip(hsv[:, :, 2] * intensity_factor, 0, 255), hsv[:, :, 2])

# Actualizar el canal V (Valor) en la imagen HSV con la intensidad aumentada
hsv[:, :, 2] = enhanced_blue.astype(np.uint8)

# Convertir la imagen HSV modificada de nuevo a BGR
enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
gray = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)

mask = cv2.inRange(hsv, lower_blue, upper_blue)

blurred_mask = cv2.GaussianBlur(mask, (3, 3), 0)

circles = cv2.HoughCircles(
    blurred_mask,
    cv2.HOUGH_GRADIENT_ALT,
    dp=1,
    minDist=14,
    param1=100,
    param2=0.48,
    minRadius=1,
    maxRadius=50
)

canvas = np.zeros_like(image)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        x, y, r = circle
        x = np.clip(x, r, gray.shape[1] - r)
        y = np.clip(y, r, gray.shape[0] - r)
        cv2.circle(canvas, (x, y), r, (0, 255, 0), 2)

        roi = gray[max(0, y - r):min(gray.shape[0], y + r), max(0, x - r):min(gray.shape[1], x + r)]
        _, roi_threshold = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # roi_gray = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        # roi_hsv = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2HSV)

        # mask_white = cv2.inRange(roi_hsv, lower_white, upper_white)
        # edges = cv2.Canny(mask_white, 300, 310, apertureSize=3)

        if cv2.countNonZero(roi_threshold) > 0:
            if roi_threshold.size > 0:
                cv2.imshow("edges", roi_threshold)
                cv2.waitKey(0) 
                cv2.destroyAllWindows()
                contours, _ = cv2.findContours(roi_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

                    if len(approx) >= 4 and len(approx) <= 10:
                        rect = cv2.minAreaRect(cnt)
                        box = cv2.boxPoints(rect)
                        box = np.int0(box)

                        angle = rect[2]

                        if angle >= -45 and angle <= 45 or angle == 90:
                            cv2.circle(image, (x, y), r, (0, 255, 0), 2)
                                

cv2.imshow("Detected Arrows", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
 
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(blurred_mask)
ax[0].set_title('Áreas Rojas detectadas')

ax[2].imshow(image)
ax[2].set_title('Áreas Rojas detectadas')


for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
