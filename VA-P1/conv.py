import numpy as np
from skimage import io
from skimage import color
import matplotlib.pyplot as plt

# Filtrado espacial - convolucion
def filterImage(inImage, kernel):
    image_height, image_width = inImage.shape
    # Asignacion dimensiones kernel, unidimensional o bidimensional
    kernel_height, kernel_width = kernel.shape if len(kernel.shape) == 2 else (1, kernel.shape[0])
    
    # Calculo de las dimensiones imagen extendida
    extra_height = image_height + kernel_height - 1
    extra_width = image_width + kernel_width - 1
    extra_image = np.zeros((extra_height, extra_width))
    # Coloco la imagen original en el centro de la matriz extendida
    extra_image[kernel_height // 2 : kernel_height // 2 + image_height, kernel_width // 2 : kernel_width // 2 + image_width] = inImage

    outImage = np.zeros((image_height, image_width))
    for i in range(image_height):
        for j in range(image_width):
            convolution_value = 0
            # Aplico el kernel en el pixel actual
            for m in range(kernel_height):
                for n in range(kernel_width):
                    # Coordenadas del área alrededor del píxel actual
                    x = i + m
                    y = j + n
                    if len(kernel.shape) == 2:
                        convolution_value += extra_image[x, y] * kernel[m, n]
                    else:
                        convolution_value += extra_image[x, y] * kernel[n] 
            outImage[i, j] = convolution_value
    
    return outImage

# Kernel gaussiano 1D
def gaussKernel1D(sigma):
    N = int(2 * np.ceil(3 * sigma) + 1)
    midpoint = N // 2
    kernel = np.zeros(N)

    for i in range(N):
        # Calculo distancia desde el centro
        x = i - midpoint
        kernel[i] = np.exp(-x**2 / (2 * sigma**2)) / (sigma * np.sqrt(2 * np.pi))

    return kernel

# Suavizado Gaussiano 2D
def gaussianFilter(inImage, sigma):
    horizontal_kernel = gaussKernel1D(sigma)
    vertical_kernel = horizontal_kernel.reshape(-1, 1)
    horizontal_filtered = filterImage(inImage, horizontal_kernel)
    outImage = filterImage(horizontal_filtered, vertical_kernel)

    return outImage

# Filtro medianas 2D
def medianFilter(inImage, filterSize):
    neighborhood = filterSize // 2
    image_height, image_width = inImage.shape
    outImage = np.zeros_like(inImage)

    for y in range(neighborhood, image_height - neighborhood):
        for x in range(neighborhood, image_width - neighborhood):
            # Cojo una sección de la imagen alrededor del píxel
            section = inImage[y - neighborhood:y + neighborhood + 1, x - neighborhood:x + neighborhood + 1]
            outImage[y, x] = np.median(section)

    return outImage


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

# ENTRADA
inImage = io.imread('imagenes-conv/gato.jpeg')
# img_input_bw = np.zeros([150,150])
# img_input_bw[60,60] = 1
# kernel = io.imread('imagenes-conv/kernel.jpg')
kernel_bw = np.array([[1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1]], dtype=np.float32) / 9.0

# BLANCO Y NEGRO
img_input_bw = black_and_white(inImage)
# kernel_bw = black_and_white(kernel)

# COMPROBACION GAUSSKERNEL1D
# sigma = 15
# kernel = gaussKernel1D(sigma)

# plt.plot(kernel, label=f'Sigma={sigma}')
# plt.xlabel('Posición')
# plt.ylabel('Valor del Kernel')
# plt.title(f'Kernel Gaussiano 1D para Sigma={sigma}')
# plt.legend()
# plt.show()

# COMPROBACION FILTERIMAGE
outImage = filterImage(img_input_bw, kernel_bw)

# COMPROBACION GAUSSIANFILTER
# sigma = 5
# outImage = gaussianFilter(img_input_bw,sigma)

# COMPROBACION MEDIANFILTER
# outImage = medianFilter(img_input_bw, 15)

# GUARDAR IMAGEN
# min = np.min(outImage)
# max = np.max(outImage)
# outImage = (outImage - min) / (max - min)
saveImage(outImage, 'imagenes-conv/saved_filterImage.jpg')


# Mostrar imágenes
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(img_input_bw, cmap='gray',vmin=0.0,vmax=1.0) 
plt.title('Imagen de entrada')
plt.subplot(1, 2, 2)
plt.imshow(outImage, cmap='gray',vmin=0.0,vmax=1.0)
plt.title('Imagen resultante')

plt.show()