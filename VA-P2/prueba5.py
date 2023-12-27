import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_blue_circles(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Definir límites para tonos de azul más amplios
    lower_blue = np.array([106, 50, 50])  # Valores inferiores para tonos azules más oscuros
    upper_blue = np.array([130, 165, 255])  # Valores superiores para tonos azules más claros

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

    blurred_mask_blue = cv2.GaussianBlur(mask_blue, (3, 3), 0)

    circles_blue = cv2.HoughCircles(
        blurred_mask_blue,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1,
        minDist=14,
        param1=100,
        param2=0.48,
        minRadius=1,
        maxRadius=50
    )

    original_image = np.copy(image)
    if circles_blue is not None:
        circles_blue = np.uint16(np.around(circles_blue))
        for circle in circles_blue[0, :]:
            x, y, r = circle
            x = np.clip(x, r, gray.shape[1] - r)
            y = np.clip(y, r, gray.shape[0] - r)

            roi = gray[max(0, y - r):min(gray.shape[0], y + r), max(0, x - r):min(gray.shape[1], x + r)]
            _, roi_threshold = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            # roi_gray = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            # roi_hsv = cv2.cvtColor(roi_gray, cv2.COLOR_BGR2HSV)

            # mask_white = cv2.inRange(roi_hsv, lower_white, upper_white)
            # edges = cv2.Canny(mask_white, 300, 310, apertureSize=3)

            if cv2.countNonZero(roi_threshold) > 0:
                if roi_threshold.size > 0:
                    # cv2.imshow("edges", roi_threshold)
                    # cv2.waitKey(0) 
                    # cv2.destroyAllWindows()
                    contours, _ = cv2.findContours(roi_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        area = cv2.contourArea(cnt)
                        peri = cv2.arcLength(cnt, True)
                        approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

                        if len(approx) >= 4 and len(approx) <= 10:
                            rect = cv2.minAreaRect(cnt)
                            angle = rect[2]

                            if angle >= -45 and angle <= 45 or angle == 90:
                                cv2.circle(original_image, (x, y), r, (0, 255, 0), 2)
                                
    return enhanced_image, original_image


def detect_red_circles(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([10, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    blurred_mask_red = cv2.GaussianBlur(mask_red, (3, 3), 0)

    circles_red = cv2.HoughCircles(
        blurred_mask_red,
        cv2.HOUGH_GRADIENT_ALT,
        dp=1,
        minDist=6,
        param1=100,
        param2=0.1, 
        minRadius=10, 
        maxRadius=50
    )

    original_image = np.copy(image)
    if circles_red is not None:
        circles_red = np.uint16(np.around(circles_red))
        for circle in circles_red[0, :]:
            x, y, r = circle
            x = np.clip(x, r, gray.shape[1] - r)
            y = np.clip(y, r, gray.shape[0] - r)
            print("cirlce")
            roi = gray[max(0, y - r):min(gray.shape[0], y + r), max(0, x - r):min(gray.shape[1], x + r)]

            _, roi_threshold = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            if cv2.countNonZero(roi_threshold) > 0:
                if roi_threshold.size > 0:
                    # cv2.imshow("edges", roi_threshold)
                    # cv2.waitKey(0) 
                    # cv2.destroyAllWindows()
                    contours, _ = cv2.findContours(roi_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:

                        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
                        aspect_ratio = w_rect / h_rect

                        if (aspect_ratio >= 1 and aspect_ratio <= 1.1) or aspect_ratio == 4.4 or aspect_ratio == 4.5:
                            area = cv2.contourArea(cnt)
                            peri = cv2.arcLength(cnt, True)
                            approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)

                            roi_signal = image[max(0, y - r):min(gray.shape[0], y + r), max(0, x - r):min(gray.shape[1], x + r)]
                            adjusted_image = cv2.convertScaleAbs(roi_signal, alpha=2.5, beta=20)
                            white_ratio, red_ratio = verificar_colores_en_roi(adjusted_image)
                            if white_ratio > 10  and red_ratio > 1 and area > 0.4 and (len(approx) == 8 or len(approx) == 10 or len(approx) == 5 or len(approx) == 7 or len(approx) == 3):
                                # cv2.imshow("edges", roi_threshold)
                                # cv2.waitKey(0) 
                                # cv2.destroyAllWindows()
                                inner_circles = cv2.HoughCircles(
                                    roi_threshold,
                                    cv2.HOUGH_GRADIENT_ALT,
                                    dp=1,
                                    minDist=2,
                                    param1=100,
                                    param2=0.3,
                                    minRadius=4,
                                    maxRadius=50
                                )

                                if inner_circles is not None:
                                    cv2.circle(original_image, (x, y), r, (0, 255, 0), 2)

    return blurred_mask_red, original_image

def verificar_colores_en_roi(roi):
    white_lower = np.array([0, 0, 200])
    white_upper = np.array([180, 30, 255])

    red_lower = np.array([0, 50, 50])
    red_upper = np.array([10, 255, 255])

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv_roi, red_lower, red_upper)
    mask_white = cv2.inRange(hsv_roi, white_lower, white_upper)

    red_pixels_percentage = np.sum(mask_red == 255) / (roi.shape[0] * roi.shape[1]) * 100
    white_pixels_percentage = np.sum(mask_white == 255) / (roi.shape[0] * roi.shape[1]) * 100

    return white_pixels_percentage, red_pixels_percentage

# Uso de la función para detectar círculos azules en la imagen
image_path = "Material Señales/00446.ppm"
detected_image_blue, image_blue = detect_blue_circles(image_path)
detected_image_red, image_red = detect_red_circles(image_blue)


# Visualizar los resultados
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(image_red, cv2.COLOR_BGR2RGB))
ax[0].set_title('Áreas azules detectadas')

# Mostrar la imagen
ax[1].imshow(detected_image_red)
ax[1].set_title('Imagen')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()