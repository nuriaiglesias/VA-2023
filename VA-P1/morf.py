import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage import color
from skimage import morphology


def erode(inImage, SE, center=[]):
    if len(center) == 0:
        center = [(SE.shape[0] // 2), (SE.shape[1] // 2)]

    P, Q = SE.shape
    M, N = inImage.shape
    outImage = np.zeros((M, N), dtype=np.uint8)

    for i in range(M):
        for j in range(N):
            match = True
            for p in range(P):
                for q in range(Q):
                    if SE[p, q] == 1:
                        x = i + p - center[0]
                        y = j + q - center[1]
                        if x < 0 or x >= M or y < 0 or y >= N:
                            match = False
                            break
                        if inImage[x, y] != 1:
                            match = False
                if not match:
                    break

            if match:
                outImage[i, j] = 1

    return outImage

def dilate(inImage, SE, center=[]):
    if len(center) == 0:
        center = [(SE.shape[0] // 2), (SE.shape[1] // 2)]

    P, Q = SE.shape
    M, N = inImage.shape
    outImage = np.zeros((M, N), dtype=np.uint8)

    for i in range(M):
        for j in range(N):
            if inImage[i, j] == 1:
                for p in range(P):
                    for q in range(Q):
                        if SE[p, q] == 1:
                            x = i + p - center[0]
                            y = j + q - center[1]
                            if x >= 0 and x < M and y >= 0 and y < N:
                                outImage[x, y] = 1

    return outImage

def opening(inImage, SE, center=[]):
    eroded = erode(inImage, SE, center)
    opened = dilate(eroded, SE, center)
    return opened

def closing(inImage, SE, center=[]):
    dilated = dilate(inImage, SE, center)
    closed = erode(dilated, SE, center)
    return closed

def hit_or_miss(inImage, objSEj, bgSE, center=[]):
    if len(center) == 0:
        center = [(objSEj.shape[0] // 2), (objSEj.shape[1] // 2)]

    if objSEj.shape != bgSE.shape:
        print("Error: elementos estructurantes incoherentes")
        return None

    if np.logical_and(objSEj, bgSE).any():
        print("Error: elementos estructurantes incoherentes")
        return None

    eroded_obj = erode(inImage, objSEj, center)
    inverted_image = 1 - inImage
    eroded_bg = erode(inverted_image, bgSE, center)
    outImage = eroded_obj * eroded_bg

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

inImage = io.imread('animo-xoel.jpeg')
inImage = black_and_white(inImage)

umbral = 0.5 
inImage_binary = (inImage > umbral).astype(float)

# binarized = np.array([[1, 0, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0, 0],
#               [0, 1, 1, 0, 0, 0],
#               [0, 1, 0, 0, 0, 0],
#               [0, 1, 0, 0, 0, 0]], dtype=np.uint8)

# SE = np.array([[1, 1]], dtype=np.uint8)

# SE = np.array([[0, 0, 0, 0, 0, 0],
#               [0, 1, 1, 1, 1, 0],
#               [0, 1, 1, 1, 1, 0],
#               [0, 1, 1, 1, 1, 0],
#               [0, 0, 0, 0, 0, 0]], dtype=np.uint8)

# SE = np.array([[1, 1, 1],
#                [0, 0, 0],
#                [1, 1, 1]])

# SE = np.array([[0, 0, 1, 1, 1, 0, 0]])

# Realizar las operaciones morfológicas
# outImageErode = erode(inImage_binary, SE)
# outImageDilate = dilate(inImage_binary, SE)
# outImageOpened = opening(inImage_binary, SE, (0,4))
# outImageClosed = closing(inImage_binary, SE)

# plt.figure(figsize=(15, 6))

# plt.subplot(2, 3, 1)
# plt.title('Imagen de entrada')
# io.imshow(inImage_binary, cmap='gray')

# plt.subplot(2, 3, 2)
# plt.title('Erosión')
# io.imshow(outImageErode, cmap='gray')

# plt.subplot(2, 3, 3)
# plt.title('Dilatación')
# io.imshow(outImageDilate, cmap='gray')

# plt.subplot(2, 3, 4)
# plt.title('Apertura')
# io.imshow(outImageOpened, cmap='gray')

# plt.subplot(2, 3, 5)
# plt.title('Cierre')
# io.imshow(outImageClosed, cmap='gray')

# plt.tight_layout()

# # Realizar la apertura y el cierre con scikit-image
# opened_image = morphology.opening(inImage_binary, footprint=SE)
# closed_image = morphology.closing(binarized, footprint=SE)

# Visualizar las imágenes
# plt.figure(figsize=(10, 5))
# plt.subplot(1, 3, 1)
# plt.title("Imagen Binarizada")
# io.imshow(binarized, cmap='gray')

# plt.subplot(1, 3, 2)
# plt.title("Apertura")
# io.imshow(opened_image, cmap='gray')

# plt.subplot(1, 3, 3)
# plt.title("Cierre")
# io.imshow(closed_image, cmap='gray')

# plt.show()

# Definir los elementos estructurantes
# objSEj = np.array([[0, 1, 0],
#                  [0, 1, 1],
#                  [0, 0, 0]])

# bgSE = np.array([[0, 0, 0],
#                 [1, 0, 0],
#                 [1, 1, 0]])

# # Crear una imagen de ejemplo
# inImage = np.array([[0, 0, 0, 0, 1, 0],
#                     [0, 1, 0, 0, 1, 1],
#                    [0, 0, 1, 0, 0, 0],
#                    [0, 0, 1, 0, 1, 0],
#                    [0, 0, 0, 1, 0, 0],
#                    [0, 0, 0, 0, 0, 0]])

# # Aplicar la transformada Hit-or-Miss
# outImage = hit_or_miss(inImage, objSEj, bgSE, center=(1,1))

# # Imprimir la imagen resultante
# print("Imagen de entrada:")
# print(inImage)
# print("\nImagen de salida (Transformada Hit-or-Miss):")
# print(outImage)