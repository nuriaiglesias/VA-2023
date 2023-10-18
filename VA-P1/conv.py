import numpy as np
from skimage import io
from skimage import color
import matplotlib.pyplot as plt

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


def medianFilter(inImage, filterSize):
    offset = filterSize // 2
    height, width = inImage.shape
    outImage = np.zeros_like(inImage)

    for y in range(offset, height - offset):
        for x in range(offset, width - offset):
            window = inImage[y - offset:y + offset + 1, x - offset:x + offset + 1]
            outImage[y, x] = np.median(window)

    return outImage



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

img_input = io.imread('khalo.jpeg')
kernel = io.imread('kernel.jpg')
# kernel_bw = np.array([[1, 1, 1, 1, 1, 1, 1],
#                                  [1, 1, 1, 1, 1, 1, 1],
#                                  [1, 1, 1, 1, 1, 1, 1],
#                                  [1, 1, 1, 1, 1, 1, 1],
#                                  [1, 1, 1, 1, 1, 1, 1],
#                                  [1, 1, 1, 1, 1, 1, 1],
#                                  [1, 1, 1, 1, 1, 1, 1]], dtype=np.float32) / 9.0

img_input_bw = black_and_white(img_input)
kernel_bw = black_and_white(kernel)

img_conv = np.zeros([50,50])
img_conv[30,30] = 1

# Comprobación gaussKernel1D
# sigma = 15
# kernel = gaussKernel1D(sigma)

# plt.plot(kernel, label=f'Sigma={sigma}')
# plt.xlabel('Posición')
# plt.ylabel('Valor del Kernel')
# plt.title(f'Kernel Gaussiano 1D para Sigma={sigma}')
# plt.legend()
# plt.show()

# Comprobación filterImage
outImage = filterImage(img_conv, kernel_bw)

# Comprobación gaussianFilter
# sigma = 15
# outImage = gaussianFilter(img_input_bw,sigma)

# Comprobación medianFilter
# outImage = medianFilter(img_input_bw, 3)

# Guardar la imagen resultante
# saveImage(outImage, 'imagen_ajustada.jpg')
# Mostrar imágenes
plt.figure()
plt.subplot(1, 2, 1)
io.imshow(img_conv, cmap='gray') 
plt.title('Imagen de entrada')
plt.subplot(1, 2, 2)
io.imshow(outImage, cmap='gray')
plt.title('Imagen resultante')

plt.show()

