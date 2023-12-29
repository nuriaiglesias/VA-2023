import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

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
    intensity_factor = 1.9  
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
        dp=0.1,
        minDist=6,
        param1=100,
        param2=0.1, 
        minRadius=10, 
        maxRadius=49
    )

    original_image = np.copy(image)
    if circles_red is not None:
        circles_red = np.uint16(np.around(circles_red))

        for circle in circles_red[0, :]:
            x, y, r = circle
            x = np.clip(x, r, gray.shape[1] - r)
            y = np.clip(y, r, gray.shape[0] - r)
            roi = gray[max(0, y - r):min(gray.shape[0], y + r), max(0, x - r):min(gray.shape[1], x + r)]
            _, roi_threshold = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if cv2.countNonZero(roi_threshold) > 0:
                if roi_threshold.size > 0:
                    contours, _ = cv2.findContours(roi_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for cnt in contours:
                        x_rect, y_rect, w_rect, h_rect = cv2.boundingRect(cnt)
                        aspect_ratio = w_rect / h_rect
                        if (aspect_ratio >= 1 and aspect_ratio <= 1.1) or aspect_ratio == 4.4 or aspect_ratio == 4.5:
                            # height, width = image.shape[:2]
                            # if int(height * 0.2) < y < int(height * 0.7) and int(width * 0.2) < x:
                                area = cv2.contourArea(cnt)
                                peri = cv2.arcLength(cnt, True)
                                approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                                roi_signal = image[max(0, y - r):min(gray.shape[0], y + r), max(0, x - r):min(gray.shape[1], x + r)]
                                adjusted_image = cv2.convertScaleAbs(roi_signal, alpha=2.3, beta=20)
                                white_ratio, red_ratio = verificar_colores_en_roi(adjusted_image)
                                if white_ratio > 20 and red_ratio > 16 and area > 0.4 and (len(approx) in [3, 5, 7, 8, 10]):
                                    inner_circles = cv2.HoughCircles(
                                        roi_threshold,
                                        cv2.HOUGH_GRADIENT_ALT,
                                        dp=0.1,
                                        minDist=3,
                                        param1=100,
                                        param2=0.35, 
                                        minRadius=4,
                                        maxRadius=19
                                    )
                                    if inner_circles is not None:
                                        inner_circles = np.uint16(np.around(inner_circles))
                                        for circle in inner_circles[0, :]:
                                            x_inner, y_inner, r_inner = circle
                                            x_inner = np.clip(x_inner, r_inner, gray.shape[1] - r_inner)
                                            y_inner = np.clip(y_inner, r_inner, gray.shape[0] - r_inner)
                                            area_inner = np.pi * (r_inner ** 2)
                                            roi_signal_inner = image[max(0, y_inner - r_inner):min(gray.shape[0], y_inner + r_inner), max(0, x_inner - r_inner):min(gray.shape[1], x_inner + r_inner)]
                                            adjusted_image_inner = cv2.convertScaleAbs(roi_signal_inner, alpha=2.5, beta=10)
                                            white_ratio, _ = verificar_colores_en_roi(adjusted_image_inner)
                                            if area_inner > 250 and white_ratio > 84:
                                                cv2.circle(original_image, (x, y), r, (0, 255, 0), 2)

    return blurred_mask_red, original_image

def verificar_colores_en_roi(roi):
    white_lower = np.array([0, 0, 220])
    white_upper = np.array([180, 50, 255])

    red_lower = np.array([0, 60, 80])
    red_upper = np.array([30, 200, 255])

    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_red = cv2.inRange(hsv_roi, red_lower, red_upper)
    mask_white = cv2.inRange(hsv_roi, white_lower, white_upper)

    red_pixels_percentage = np.sum(mask_red == 255) / (roi.shape[0] * roi.shape[1]) * 100
    white_pixels_percentage = np.sum(mask_white == 255) / (roi.shape[0] * roi.shape[1]) * 100

    return white_pixels_percentage, red_pixels_percentage

def detect_red_triangles(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 130, 20])
    upper_red = np.array([10, 255, 255])
    lower_red_strange = np.array([169, 94, 20])
    upper_red_strange = np.array([179, 255, 255])

    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    mask_red_strange = cv2.inRange(hsv, lower_red_strange, upper_red_strange)

    combined_mask = cv2.bitwise_or(mask_red, mask_red_strange)
    blurred_mask_red = cv2.GaussianBlur(combined_mask, (3, 3), 0)
    _, blurred_threshold = cv2.threshold(blurred_mask_red, 0, 255, cv2.THRESH_BINARY )

    contours, _ = cv2.findContours(blurred_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    original_image = image.copy()

    for cnt in contours:
        epsilon = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(cnt)

        if (len(approx) == 3 or len(approx) == 7 or len(approx) == 6 or len(approx) == 5) and area > 610:
            x, y, w, h = cv2.boundingRect(cnt)
            roi_signal = image[y:y+h, x:x+w]
            adjusted_image = cv2.convertScaleAbs(roi_signal, alpha=1.5, beta=1)
            white_ratio, red_ratio = verificar_colores_en_roi(adjusted_image)
            if white_ratio > 20 and white_ratio <= 70 and red_ratio > 10 and red_ratio < 40:
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    return blurred_threshold, original_image

def detect_blue_square(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 140, 50])
    upper_blue = np.array([124, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    blurred_mask_blue = cv2.GaussianBlur(mask_blue, (3, 3), 0)

    original_image = np.copy(image)
    
    _, roi_threshold = cv2.threshold(blurred_mask_blue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(roi_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    original_image = np.copy(image)

    for cnt in contours:
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(cnt)

        if (len(approx) == 4 or len(approx) == 7 or len(approx) == 5) and area > 50:  #1200
            x, y, w, h = cv2.boundingRect(cnt)
            _, roi_threshold_internal = cv2.threshold(mask_blue[y:y+h, x:x+w], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            roi_threshold_internal = cv2.GaussianBlur(roi_threshold_internal, (5, 5), 0)
            internal_contours, _ = cv2.findContours(roi_threshold_internal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for internal_cnt in internal_contours:
                internal_approx = cv2.approxPolyDP(internal_cnt, 0.04 * cv2.arcLength(internal_cnt, True), True)
                if len(internal_approx) == 4 or len(internal_approx) == 7 or len(internal_approx) == 3 or (len(approx) >= 1 and len(approx) <= 20):
                    small_roi = image[y + h // 4:y + 3 * h // 4, x + w // 4:x + 3 * w // 4]
                    small_roi_hsv = cv2.cvtColor(small_roi, cv2.COLOR_BGR2HSV)
            
                    lower_black = np.array([0, 0, 0])
                    upper_black = np.array([240, 50, 80])

                    mask_black = cv2.inRange(small_roi_hsv, lower_black, upper_black)
                    
                    if cv2.countNonZero(mask_black) > 0:
                        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(original_image, "Indicacion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return blurred_mask_blue, original_image

image_path = "Material Señales/00023.ppm"
# 235,262,001
# fallan 262, 159, 219
detected_image_blue, image_blue = detect_blue_circles(image_path)
detected_image_red, image_red = detect_red_circles(image_blue)
detected_image_red_triangles, image_red_triangles = detect_red_triangles(image_red)
detected_image_blue_square, image_blue_square = detect_blue_square(image_red_triangles)

# Visualizar los resultados
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(image_blue_square, cv2.COLOR_BGR2RGB))
ax[0].set_title('Áreas azules detectadas')

# Mostrar la imagen
ax[1].imshow(detected_image_blue_square)
ax[1].set_title('Imagen')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()