import numpy as np
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
import math


def filterImage(inImage, kernel):
    image_height, image_width = inImage.shape
    kernel_height, kernel_width = kernel.shape if len(kernel.shape) == 2 else (1, kernel.shape[0])
    
    extra_height = image_height + kernel_height - 1
    extra_width = image_width + kernel_width - 1
    extra_image = np.zeros((extra_height, extra_width))
    extra_image[kernel_height // 2 : kernel_height // 2 + image_height, kernel_width // 2 : kernel_width // 2 + image_width] = inImage

    outImage = np.zeros((image_height, image_width))
    for i in range(image_height):
        for j in range(image_width):
            convolution_value = 0
            for m in range(kernel_height):
                for n in range(kernel_width):
                    x = i + m
                    y = j + n
                    if len(kernel.shape) == 2:
                        convolution_value += extra_image[x, y] * kernel[m, n]
                    else:
                        convolution_value += extra_image[x, y] * kernel[n] 
            outImage[i, j] = convolution_value
    
    return outImage


def gaussKernel1D(sigma):
    N = int(2 * np.ceil(3 * sigma) + 1)
    midpoint = N // 2
    kernel = np.zeros(N)

    for i in range(N):
        x = i - midpoint
        kernel[i] = np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    return kernel

def gaussianFilter(inImage, sigma):
    horizontal_kernel = gaussKernel1D(sigma)
    vertical_kernel = horizontal_kernel.reshape(-1, 1)
    horizontal_filtered = filterImage(inImage, horizontal_kernel)
    outImage = filterImage(horizontal_filtered, vertical_kernel)

    return outImage

# Gradiente de una imagen
def gradientImage(inImage, operator):
    if operator == 'Roberts':
        gx_kernel = np.array([[-1, 0], [0, 1]])
        gy_kernel = np.array([[0, -1], [1, 0]])
    elif operator == 'CentralDiff':
        gx_kernel = np.array([[-1, 0, 1]])
        gy_kernel = gx_kernel.T
    elif operator == 'Prewitt':
        gx_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        gy_kernel = gx_kernel.T 
    elif operator == 'Sobel':
        gx_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        gy_kernel = gx_kernel.T
    else:
        raise ValueError("Operador proporcionado invalido. Por favor, seleccione entre los siguientes: 'Roberts', 'CentralDiff', 'Prewitt' o 'Sobel'.")

    gx = filterImage(inImage, gx_kernel)
    gy = filterImage(inImage, gy_kernel)
    
    return gx, gy

# Filtro Laplaciano de Gaussiano
def LoG(inImage, sigma):
    log_kernel = np.array([[-1, -1, -1],
                 [-1, 8, -1],
                 [-1, -1, -1]])
    gaussian_output = gaussianFilter(inImage,sigma)
    log_output = filterImage(gaussian_output, log_kernel)
    return log_output

# Detector de bordes de canny
def edgeCanny(inImage, sigma, tlow, thigh):
    gaussian_output = gaussianFilter(inImage, sigma)

    gx, gy = gradientImage(gaussian_output, 'Sobel')

    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)

    # Supresión no maxima
    image_height, image_width = inImage.shape
    delineated_edges = np.zeros((image_height, image_width), dtype=np.float32)

    # Comprobacion del anterior y posterior pixel en la misma direccion y sus valores
    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            angle = math.degrees(direction[i, j]) 
            if(angle < 0):
                angle += 180
            # FILA
            if(angle >= 0 and angle < 22.5) or (angle >= 157.5 and angle <= 180):
                if (magnitude[i, j] >= magnitude[i, j + 1]) and (magnitude[i, j] >= magnitude[i, j - 1]):
                    delineated_edges[i, j] = magnitude[i, j]
                else: 
                    delineated_edges[i, j] = 0
            # DIAGONAL DELANTE
            elif(angle >= 22.5 and angle < 67.5):
                if (magnitude[i, j] >= magnitude[i + 1, j + 1]) and (magnitude[i, j] >= magnitude[i - 1, j - 1]):
                    delineated_edges[i, j] = magnitude[i, j]
                else: 
                    delineated_edges[i, j] = 0
            # DIAGONAL DETRAS
            elif(angle >= 112.5 and angle < 157.5):
                if (magnitude[i, j] >= magnitude[i + 1, j - 1]) and (magnitude[i, j] >= magnitude[i - 1, j + 1]):
                    delineated_edges[i, j] = magnitude[i, j]
                else: 
                    delineated_edges[i, j] = 0
            # COLUMNA
            elif(angle >= 67.5 and angle < 112.5):
                if (magnitude[i, j] >= magnitude[i - 1, j]) and (magnitude[i, j] >= magnitude[i + 1, j]):
                    delineated_edges[i, j] = magnitude[i, j]
                else: 
                    delineated_edges[i, j] = 0

    # Umbralizacion por histeresis
    strong_edges = delineated_edges > thigh
    edge_image = np.zeros((image_height, image_width), dtype=np.float32)
    # Registro de los pixeles visitados
    visited = np.zeros((image_height, image_width), dtype=bool)

    def detect_edges(current_row, current_col):
        if current_row < 0 or current_row >= image_height or current_col < 0 or current_col >= image_width or visited[current_row, current_col]:
            return
        # Si supera el umbral bajo, se marca como fuerte
        if delineated_edges[current_row, current_col] > tlow:
            edge_image[current_row, current_col] = 1
            visited[current_row, current_col] = True

            # Recorro los 8 pixeles adyacentes
            for adj_row in range(-1, 2):
                for adj_col in range(-1, 2):
                    # Coordenadas del pixel adyacente
                    adjacent_x, adjacent_y = current_row + adj_row, current_col + adj_col
                    if 0 <= adjacent_x < image_height and 0 <= adjacent_y < image_width and not visited[adjacent_x, adjacent_y]:
                        # Angulos del gradiente del pixel adyacente y del actual
                        adj_angle = math.degrees(direction[adjacent_x, adjacent_y])
                        current_angle = math.degrees(direction[current_row, current_col])
                        if adj_angle < 0:
                            adj_angle += 180
                        if current_angle < 0:
                            current_angle += 180
                        angle_difference = abs(current_angle - adj_angle)
                        if angle_difference < 45 or (angle_difference > 135 and angle_difference < 180):
                            # Llamada recursiva para buscar bordes débiles en esa dirección
                            detect_edges(adjacent_x, adjacent_y)


    # Conecta bordes débiles adyacentes a bordes fuertes
    for current_row in range(image_height):
        for current_col in range(image_width):
            if strong_edges[current_row, current_col] and not visited[current_row, current_col]:
                detect_edges(current_row, current_col)

    return edge_image

# Detector de esquinas basado en SUSAN
def cornerSusan(inImage, r, t):
    outCorners = np.zeros_like(inImage)
    usanArea = np.zeros_like(inImage)
    image_height, image_width = inImage.shape[:2]
    
    # Creacion mascara circular
    circle_mask = np.zeros((2*r+1, 2*r+1))
    center = (r, r)
    for i in range(2*r+1):
        for j in range(2*r+1):
            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            if dist <= r:
                circle_mask[i, j] = 1
    
    # Calculo del maximo valor de intensidades de la máscara
    max_usan = np.sum(circle_mask) 
    # Umbral para considerar esquinas
    g = 3/4 * max_usan  
    
    # Iteracion sobre la imagen para detectar esquinas
    for y in range(r, image_height - r):
        for x in range(r, image_width - r):
            # Obtencion de la intensidad del pixel (y, x)
            pixel_intensity = inImage[y, x]
            # Define una región centrada en (y, x)
            region = inImage[y-r:y+r+1, x-r:x+r+1]
            # Diferencia entre la intensidad del píxel central y los píxeles de la región
            diff = np.abs(region - pixel_intensity)
            # Determinacion de diff en los píxeles dentro de la forma circular
            susan_values = circle_mask * (diff <= t)
            
            # Suma los pixeles dentro de la región que cumplen con la condición
            usan = np.sum(susan_values)
            # Normaliza
            usanArea[y, x] = usan / max_usan 
            
            # Se resalta la esquina si cumple la condición
            if usan < g:
                outCorners[y, x] = g - usan

    return outCorners, usanArea

def black_and_white(img):
    if len(img.shape) == 2:
        min = np.min(img)
        max = np.max(img)
        img = (img - min) / (max - min)
        img = img.astype(float)
    elif len(img.shape) == 3:
        if img.shape[2] == 4:  
            img = img[:, :, :3] 
        img = color.rgb2gray(img).astype(float)
    return img

def saveImage(image, filename):
    scaled_image = (image * 255).astype(np.uint8)
    io.imsave(filename, scaled_image)

inImage = io.imread('girasol.jpeg')

inImage = black_and_white(inImage)

# gx, gy = gradientImage(inImage, 'Prewitt')
# outImage = LoG(inImage, 0.5)
outImage = edgeCanny(inImage, 0.3, 0.1, 0.8)
# radius = 10
# threshold = 0.6
# corners, usan_area = cornerSusan(inImage, radius, threshold)

# saveImage(outImage, 'imagenes-bordes/imagen_guardada_circle12.jpg')

# Visualizar cornerSusan
# plt.figure(figsize=(8, 6))

# plt.subplot(1, 3, 1)
# plt.imshow(inImage, cmap='gray')
# plt.title('Imagen original')
# plt.axis('off')

# plt.subplot(1, 3, 2)
# plt.imshow(corners, cmap='gray')
# plt.title('Mapa de esquinas')
# plt.axis('off')

# plt.subplot(1, 3, 3)
# plt.imshow(usan_area, cmap='gray')
# plt.title('usanArea')
# plt.axis('off')

# plt.tight_layout()
# plt.show()

# Visualizar 
plt.figure()
plt.subplot(1, 2, 1)
io.imshow(inImage, cmap='gray') 
plt.title('Imagen de entrada')
plt.subplot(1, 2, 2)
io.imshow(outImage, cmap='gray')
plt.title('Imagen resultante')
plt.show()