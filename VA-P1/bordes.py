import numpy as np
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
import math
from skimage import measure



def filterImage(inImage, kernel):
    height, width = inImage.shape
    k_height, k_width = kernel.shape if len(kernel.shape) == 2 else (1, kernel.shape[0])
    
    extended_height = height + k_height - 1
    extended_width = width + k_width - 1
    extendedImage = np.zeros((extended_height, extended_width))    
    extendedImage[k_height // 2 : k_height // 2 + height, k_width // 2 : k_width // 2 + width] = inImage

    outImage = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            pixel_result = 0            
            for m in range(k_height):
                for n in range(k_width):
                    x = i + m
                    y = j + n
                    if len(kernel.shape) == 2:
                        pixel_result += extendedImage[x, y] * kernel[m, n]
                    else:
                        pixel_result += extendedImage[x, y] * kernel[n]            
            outImage[i, j] = pixel_result
    
    return outImage


def gaussKernel1D(sigma):
    N = int(2 * np.ceil(3 * sigma) + 1)
    center = N // 2
    kernel = np.zeros(N)

    for i in range(N):
        x = i - center
        kernel[i] = np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    return kernel

def gaussianFilter(inImage, sigma):
    kernel_x = gaussKernel1D(sigma)
    kernel_y = kernel_x.reshape(-1, 1)
    intermediate_image = filterImage(inImage, kernel_x)
    outImage = filterImage(intermediate_image, kernel_y)

    return outImage

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
        raise ValueError("Operador no válido. Debe ser 'Roberts', 'CentralDiff', 'Prewitt' o 'Sobel'.")

    gx = filterImage(inImage, gx_kernel)
    gy = filterImage(inImage, gy_kernel)
    
    return gx, gy

def LoG(inImage, sigma):
    laplacian_kernel = np.array([[-1, -1, -1],
                 [-1, 8, -1],
                 [-1, -1, -1]])
    gaussian_image = gaussianFilter(inImage,sigma)
    laplacian_result = filterImage(gaussian_image, laplacian_kernel)
    return laplacian_result

def edgeCanny(inImage, sigma, tlow, thigh):
    smoothed_image = gaussianFilter(inImage, sigma)

    gx, gy = gradientImage(smoothed_image, 'Sobel')

    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)

    # Supresión no máxima
    height, width = inImage.shape
    suppressed_image = np.zeros((height, width), dtype=np.float32)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            angle = math.degrees(direction[i, j]) 
            if(angle < 0):
                angle += 180
            # print(angle)
            # FILA
            if(angle >= 0 and angle < 22.5) or (angle >= 157.5 and angle <= 180):
                if (magnitude[i, j] >= magnitude[i, j + 1]) and (magnitude[i, j] >= magnitude[i, j - 1]):
                    suppressed_image[i, j] = magnitude[i, j]
                else: 
                    suppressed_image[i, j] = 0
            # DIAGONAL DELANTE
            elif(angle >= 22.5 and angle < 67.5):
                if (magnitude[i, j] >= magnitude[i + 1, j + 1]) and (magnitude[i, j] >= magnitude[i - 1, j - 1]):
                    suppressed_image[i, j] = magnitude[i, j]
                else: 
                    suppressed_image[i, j] = 0
            # DIAGONAL DETRAS
            elif(angle >= 112.5 and angle < 157.5):
                if (magnitude[i, j] >= magnitude[i + 1, j - 1]) and (magnitude[i, j] >= magnitude[i - 1, j + 1]):
                    suppressed_image[i, j] = magnitude[i, j]
                else: 
                    suppressed_image[i, j] = 0
            # COLUMNA
            elif(angle >= 67.5 and angle < 112.5):
                if (magnitude[i, j] >= magnitude[i - 1, j]) and (magnitude[i, j] >= magnitude[i + 1, j]):
                    suppressed_image[i, j] = magnitude[i, j]
                else: 
                    suppressed_image[i, j] = 0

    #Umbralización por histéresis
    strong_edges = (suppressed_image > thigh)
    weak_edges = (suppressed_image >= tlow) & (suppressed_image <= thigh)
    # weakest_edges = (suppressed_image < tlow)
    suppressed_image = np.where(strong_edges, 1, suppressed_image)
    # suppressed_image = np.where(weakest_edges, 0, suppressed_image)
    while True:
        prev_suppressed = np.copy(suppressed_image)
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if weak_edges[i, j]:
                    neighbors = [suppressed_image[i - 1, j], suppressed_image[i + 1, j], suppressed_image[i + 1, j - 1], suppressed_image[i - 1, j + 1], suppressed_image[i + 1, j + 1], suppressed_image[i - 1, j - 1], suppressed_image[i, j - 1],suppressed_image[i, j + 1]]
                    max_neighbor_value = max(neighbors)
                    if max_neighbor_value == 1:
                        suppressed_image[i, j] = 1

        strong_edges = (suppressed_image > thigh)
        weak_edges = (suppressed_image >= tlow) & (suppressed_image <= thigh)

        if np.array_equal(prev_suppressed, suppressed_image):
            break
    suppressed_image[suppressed_image < 1] = 0
    return suppressed_image

def cornerSusan(inImage, r, t):
    outCorners = np.zeros_like(inImage)
    usanArea = np.zeros_like(inImage)
    
    height, width = inImage.shape[:2]

    mask = np.zeros((2*r+1, 2*r+1))
    center = (r, r)
    for i in range(2*r+1):
        for j in range(2*r+1):
            dist = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            if dist <= r:
                mask[i, j] = 1

    max_usan = np.sum(mask) 

    for y in range(r, height - r):
        for x in range(r, width - r):
            core_intensity = inImage[y, x]

            region = inImage[y-r:y+r+1, x-r:x+r+1]

            diff = np.abs(region - core_intensity)

            usan = np.sum(mask * (diff <= t))

            usanArea[y, x] = usan / max_usan 

            if usan / max_usan < t:
                outCorners[y, x] = 1  

    return outCorners, usanArea

def black_and_white(img):
    if len(img.shape) == 2:
        img = img.astype(float) / 255.0
    elif len(img.shape) == 3:
        if img.shape[2] == 4:  
            img = img[:, :, :3] 
        img = color.rgb2gray(img).astype(float)
    return img

def saveImage(image, filename):
    scaled_image = (image * 255).astype(np.uint8)
    io.imsave(filename, scaled_image)

inImage = io.imread('imagenes-bordes/circles1.png')

inImage = black_and_white(inImage)

# gx, gy = gradientImage(inImage, 'Prewitt')

# outImage = LoG(inImage, 0.5)
# se me va hacia arriba, se me va hacia abajo; se me va hacia la izq
# region_of_interest = inImage[170:200, 180:210]

outImage = edgeCanny(inImage, 0.3, 0.1, 0.8)

# radius = 10
# threshold = 0.6
# corners, usan_area = cornerSusan(inImage, radius, threshold)


# saveImage(outImage, 'imagenes-bordes/imagen_guardada_circle12.jpg')
# Visualizar el mapa de esquinas
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
plt.figure()
plt.subplot(1, 2, 1)
io.imshow(inImage, cmap='gray') 
plt.title('Imagen de entrada')
plt.subplot(1, 2, 2)
io.imshow(outImage, cmap='gray')
plt.title('Imagen resultante')
# plt.subplot(1, 3, 3)
# plt.imshow(gy, cmap='gray')
# plt.title('gy')
plt.show()