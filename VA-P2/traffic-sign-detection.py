import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

def detect_blue_circles(image):
    global counter_o
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

            if cv2.countNonZero(roi_threshold) > 0:
                contours, _ = cv2.findContours(roi_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                    if (len(approx) == 4 or len(approx) == 5 or len(approx) == 6) and (70 > area):
                        tipo_señal = "Obligacion"
                        if not os.path.exists(os.path.join(base_dir, number_image, tipo_señal)):
                            os.makedirs(os.path.join(base_dir, number_image, tipo_señal))
                        average_signal_size = np.mean([signal[2] for signal in detected_signals]) if detected_signals else 0
                        threshold_distance = 0.2 * average_signal_size
                        if any(np.linalg.norm(np.array([x, y]) - np.array(signal[:2])) < threshold_distance for signal in detected_signals):
                            continue
                        detected_signals.append([x, y, r]) 
                        cv2.circle(original_image, (x, y), r, (0, 255, 0), 2)
                        cv2.putText(original_image, tipo_señal, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        counter_o += 1
                        roi_image = image[max(0, y - r):min(image.shape[0], y + r), max(0, x - r):min(image.shape[1], x + r)]
                        detect_signals_and_save(roi_image, tipo_señal, counter_o)

    return blurred_mask_blue, original_image


def detect_red_circles(image):
    global counter_ph
    circles_red = cv2.HoughCircles(
        blurred_mask_red_circles,
        cv2.HOUGH_GRADIENT_ALT,
        dp=0.1,
        minDist=6,
        param1=100,
        param2=0.5, 
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
                contours, _ = cv2.findContours(roi_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    area = cv2.contourArea(cnt)
                    peri = cv2.arcLength(cnt, True)
                    approx = cv2.approxPolyDP(cnt, 0.03 * peri, True)
                    roi_signal = image[max(0, y - r):min(image.shape[0], y + r), max(0, x - r):min(image.shape[1], x + r)]
                    adjusted_image = cv2.convertScaleAbs(roi_signal, alpha=2.3, beta=20)
                    white_ratio, red_ratio = verificar_colores_en_roi(adjusted_image)
                    if (70 > white_ratio > 21) and (60 > red_ratio >= 25) and (850 > area >= 30) and (len(approx) in [1, 4, 8] or (len(approx) >= 6 and len(approx) <= 6.05)):
                        inner_circles = cv2.HoughCircles(
                            roi_threshold,
                            cv2.HOUGH_GRADIENT,
                            dp=1,
                            minDist=2000,
                            param1=100,
                            param2=4,
                            minRadius=9,
                            maxRadius=45
                        )
                        if inner_circles is not None:
                            inner_circles = np.uint16(np.around(inner_circles))
                            for circle in inner_circles[0, :]:
                                x_inner, y_inner, r_inner = circle
                                x_inner = np.clip(x_inner, r_inner, gray.shape[1] - r_inner)
                                y_inner = np.clip(y_inner, r_inner, gray.shape[0] - r_inner)
                                area_inner = np.pi * (r_inner ** 2)
                                roi_signal_inner = image[max(0, y_inner - r_inner):min(image.shape[0], y_inner + r_inner), max(0, x_inner - r_inner):min(image.shape[1], x_inner + r_inner)]
                                adjusted_image_inner = cv2.convertScaleAbs(roi_signal_inner, alpha=2.5, beta=10)
                                white_ratio, red_ratio = verificar_colores_en_roi(adjusted_image_inner)
                                if (3800 > area_inner > 450) and (105 > white_ratio > 84):
                                    tipo_señal = "Prohibicion"
                                    if not os.path.exists(os.path.join(base_dir, number_image, tipo_señal)):
                                        os.makedirs(os.path.join(base_dir, number_image, tipo_señal))
                                    average_signal_size = np.mean([signal[2] for signal in detected_signals]) if detected_signals else 0
                                    threshold_distance = 0.2 * average_signal_size
                                    if any(np.linalg.norm(np.array([x, y]) - np.array(signal[:2])) < threshold_distance for signal in detected_signals):
                                        continue
                                    detected_signals.append([x, y, r]) 
                                    cv2.rectangle(original_image, (x - r, y - r), (x + r, y + r), (0, 255, 0), 2)
                                    cv2.putText(original_image, tipo_señal, (x - 30, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    counter_ph += 1
                                    roi_image = image[max(0, y - r):min(image.shape[0], y + r), max(0, x - r):min(image.shape[1], x + r)]
                                    detect_signals_and_save(roi_image, tipo_señal, counter_ph)

    return blurred_mask_red_circles, original_image

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
    global counter_ph, counter_pl
    _, blurred_threshold = cv2.threshold(blurred_mask_red_triangles, 0, 255, cv2.THRESH_BINARY)
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
                red_upper, red_lower = contar_pixeles_rojos(adjusted_image)
                cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                average_signal_size = np.mean([signal[2] for signal in detected_signals]) if detected_signals else 0
                threshold_distance = 0.2 * average_signal_size
                if any(np.linalg.norm(np.array([x, y]) - np.array(signal[:2])) < threshold_distance for signal in detected_signals):
                    continue
                r = len(approx) // 3
                detected_signals.append([x, y, r]) 
                if red_lower > red_upper:
                    tipo_señal1 = "Prohibicion"
                    cv2.putText(original_image, tipo_señal1, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if not os.path.exists(os.path.join(base_dir, number_image, tipo_señal1)):
                        os.makedirs(os.path.join(base_dir, number_image, tipo_señal1))
                    counter_ph += 1
                    detect_signals_and_save(roi_signal, tipo_señal1, counter_ph)
                else:
                    tipo_señal2 = "Peligro"
                    cv2.putText(original_image, tipo_señal2, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if not os.path.exists(os.path.join(base_dir, number_image, tipo_señal2)):
                        os.makedirs(os.path.join(base_dir, number_image, tipo_señal2))
                    counter_pl += 1
                    detect_signals_and_save(roi_signal, tipo_señal2, counter_pl)

    return blurred_threshold, original_image

def contar_pixeles_rojos(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

    altura, _ = binary.shape[:2]
    parte_superior = binary[0:altura//2, :]
    parte_inferior = binary[altura//2:altura, :]

    red_upper = cv2.countNonZero(parte_superior)
    red_lower = cv2.countNonZero(parte_inferior)

    return red_upper, red_lower


def detect_blue_square(image):   
    global counter_i 
    _, roi_threshold = cv2.threshold(blurred_mask_blue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(roi_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    original_image = np.copy(image)
    for cnt in contours:
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        area = cv2.contourArea(cnt)

        if (len(approx) == 4 or len(approx) == 7 or len(approx) == 5) and area > 50: 
            x, y, w, h = cv2.boundingRect(cnt)
            roi_signal = image[y:y+h, x:x+w]
            _, roi_threshold_internal = cv2.threshold(mask_blue[y:y+h, x:x+w], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            roi_threshold_internal = gaussian(roi_threshold_internal)
            internal_contours, _ = cv2.findContours(roi_threshold_internal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for internal_cnt in internal_contours:
                internal_approx = cv2.approxPolyDP(internal_cnt, 0.04 * cv2.arcLength(internal_cnt, True), True)
                if len(internal_approx) == 4 or len(internal_approx) == 7 or len(internal_approx) == 3 or len(approx) == 5:
                    small_roi = image[y + h // 4:y + 3 * h // 4, x + w // 4:x + 3 * w // 4]
                    small_roi_hsv = cv2.cvtColor(small_roi, cv2.COLOR_BGR2HSV)
            
                    lower_black = np.array([0, 0, 0])
                    upper_black = np.array([240, 50, 80])

                    mask_black = cv2.inRange(small_roi_hsv, lower_black, upper_black)
                    
                    if cv2.countNonZero(mask_black) > 0:
                        tipo_señal = "Indicacion"
                        if not os.path.exists(os.path.join(base_dir, number_image, tipo_señal)):
                            os.makedirs(os.path.join(base_dir, number_image, tipo_señal))
                        average_signal_size = np.mean([signal[2] for signal in detected_signals]) if detected_signals else 0
                        threshold_distance = 0.2 * average_signal_size
                        if any(np.linalg.norm(np.array([x, y]) - np.array(signal[:2])) < threshold_distance for signal in detected_signals):
                            continue
                        r = len(approx) // 3
                        detected_signals.append([x, y, r]) 
                        cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(original_image, "Indicacion", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        counter_i += 1
                        detect_signals_and_save(roi_signal, "Indicacion", counter_i)


    return blurred_mask_blue, original_image

def gaussian(mask):
    return cv2.GaussianBlur(mask, (3, 3), 0)

def preprocess_image(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Values for blue
    lower_blue = np.array([106, 120, 50])
    upper_blue = np.array([130, 255, 255])

    # Values for red
    lower_red = np.array([0, 125, 40])
    upper_red = np.array([10, 255, 255])
    lower_red_strange = np.array([175, 100, 40])
    upper_red_strange = np.array([200, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    blurred_mask_blue = gaussian(mask_blue)

    mask_red_triangles = cv2.inRange(hsv, lower_red, upper_red)
    mask_red_strange = cv2.inRange(hsv, lower_red_strange, upper_red_strange)

    combined_mask = cv2.bitwise_or(mask_red_triangles, mask_red_strange)
    blurred_mask_red_triangles = gaussian(combined_mask)
 

    mask_red_circles = cv2.inRange(hsv, lower_red, upper_red)
    blurred_mask_red_circles = gaussian(mask_red_circles)

    return (
        mask_blue, blurred_mask_blue, blurred_mask_red_triangles, blurred_mask_red_circles, gray
    )

def detect_signals_and_save(roi, signal_type, counter):
    file_path = os.path.join(base_dir, number_image, signal_type, f"{signal_type}_detected{counter}.jpg")
    cv2.imwrite(file_path, roi)

base_dir = "Detected_signal"
number_image = "00001.ppm"
image_path = "Material Señales/" + number_image
detected_signals = []
global counter_o, counter_ph, counter_pl, counter_i
counter_o = 0
counter_ph = 0
counter_pl = 0
counter_i = 0

detected_signal_types = ["Indicacion", "Prohibicion", "Peligro", "Obligacion"]

for signal_type in detected_signal_types:
    folder_path = os.path.join(base_dir, number_image, signal_type)
    if os.path.exists(folder_path) and os.listdir(folder_path):
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            os.remove(file_path)

image = cv2.imread(image_path)
(mask_blue, blurred_mask_blue, blurred_mask_red_triangles, blurred_mask_red_circles, gray) = preprocess_image(image)

detected_image_blue, image_blue = detect_blue_circles(image)
detected_image_blue_square, image_blue_square = detect_blue_square(image_blue)
detected_image_red, image_red = detect_red_circles(image_blue_square)
detected_image_red_triangles, image_red_triangles = detect_red_triangles(image_red)

# plt.imshow(cv2.cvtColor(image_red_triangles, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.title('Result')
# plt.show()

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(cv2.cvtColor(image_red_triangles, cv2.COLOR_BGR2RGB))
ax[0].set_title('Imagen resultante')

ax[1].imshow(blurred_mask_red_circles)
ax[1].set_title('Áreas rojas detectadas')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()