import numpy as np
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from scipy.signal import convolve2d


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
        # Operador de Roberts
        gx_kernel = np.array([[-1, 0], [0, 1]])
        gy_kernel = np.array([[0, -1], [1, 0]])
    elif operator == 'CentralDiff':
        # Operador de Diferencias Centrales
        gx_kernel = np.array([-1, 0, 1])
        gy_kernel = gx_kernel.reshape(1, -1)
    elif operator == 'Prewitt':
        # Operador de Prewitt
        gx_kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        gy_kernel = gx_kernel.T 
    elif operator == 'Sobel':
        # Operador de Sobel
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

inImage = io.imread('circles.png')

inImage = black_and_white(inImage)

# gx, gy = gradientImage(inImage, 'Roberts')

outImage = LoG(inImage, 10)

plt.figure()
plt.subplot(1, 2, 1)
io.imshow(inImage, cmap='gray') 
plt.title('Imagen de entrada')
plt.subplot(1, 2, 2)
io.imshow(outImage, cmap='gray')
plt.title('Imagen resultante')

plt.show()