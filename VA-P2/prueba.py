import cv2
import numpy as np

# Cargar la imagen
image = cv2.imread("Material-Señales/00446.ppm")  # Cambiar 'ruta_de_la_imagen.jpg' por la ruta de tu imagen

# Convertir la imagen a escala de grises
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Aplicar umbralización para binarizar la imagen
_, binary_image = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)

# Aplicar operaciones morfológicas para limpiar y delinear objetos
kernel = np.ones((3,3), np.uint8)
morph_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
morph_image = cv2.morphologyEx(morph_image, cv2.MORPH_GRADIENT, kernel)

# Encontrar contornos
contours, _ = cv2.findContours(morph_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Detección de círculos en los contornos encontrados
detected_circles = []

for contour in contours:
    area = cv2.contourArea(contour)
    if area > 50:  # Ajusta el área mínima para considerar como círculo
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.3 * perimeter, True)
        num_sides = len(approx)
        if num_sides >= 1 and num_sides <= 3:  # Puede ajustar el número de lados permitidos para identificar círculos
            (x, y), radius = cv2.minEnclosingCircle(contour)
            detected_circles.append((int(x), int(y), int(radius)))

# Dibujar círculos detectados en la imagen original
for circle in detected_circles:
    cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 4)
    cv2.rectangle(image, (circle[0] - 5, circle[1] - 5), (circle[0] + 5, circle[1] + 5), (0, 128, 255), -1)
    print("Column Number:", circle[0])
    print("Row Number:", circle[1])
    print("Radius is:", circle[2])

# Mostrar la imagen con círculos detectados
cv2.imshow('Detected Circles', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
