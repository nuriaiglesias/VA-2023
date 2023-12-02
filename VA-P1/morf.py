import numpy as np
from skimage import io
import matplotlib.pyplot as plt
from skimage import color

# Operador morfologico: erosion
def erode(inImage, SE, center=[]):
    if len(center) == 0:
        center = [(SE.shape[0] // 2), (SE.shape[1] // 2)]

    se_rows, se_cols = SE.shape
    input_rows, input_cols = inImage.shape
    outImage = np.zeros((input_rows, input_cols), dtype=np.float32)

    for i in range(input_rows):
        for j in range(input_cols):
            concordance = True
            for p in range(se_rows):
                for q in range(se_cols):
                    if SE[p, q] == 1:
                        # Calculo de las coordenadas de la posicion del pixel del SE en la imagen input
                        SE_in_input_x = i + p - center[0]
                        SE_in_input_y = j + q - center[1]
                        if SE_in_input_x < 0 or SE_in_input_x >= input_rows or SE_in_input_y < 0 or SE_in_input_y >= input_cols:
                            concordance = False
                            break
                        if inImage[SE_in_input_x, SE_in_input_y] != 1:
                            concordance = False
                if not concordance:
                    break
            # Si todos los pixeles coinciden
            if concordance:
                outImage[i, j] = 1

    return outImage

# Operador morfologico: dilatacion
def dilate(inImage, SE, center=[]):
    if len(center) == 0:
        center = [(SE.shape[0] // 2), (SE.shape[1] // 2)]

    se_rows, se_cols = SE.shape
    input_rows, input_cols = inImage.shape
    outImage = np.zeros((input_rows, input_cols), dtype=np.float32)

    # Dilata si un pixel activo de la imagen coincide con uno activo en el SE
    for i in range(input_rows):
        for j in range(input_cols):
            if inImage[i, j] == 1:
                for p in range(se_rows):
                    for q in range(se_cols):
                        if SE[p, q] == 1:
                            SE_in_input_x = i + p - center[0]
                            SE_in_input_y = j + q - center[1]
                            if SE_in_input_x >= 0 and SE_in_input_x < input_rows and SE_in_input_y >= 0 and SE_in_input_y < input_cols:
                                outImage[SE_in_input_x, SE_in_input_y] = 1

    return outImage

# Operador morfologico: abertura
def opening(inImage, SE, center=[]):
    after_erode = erode(inImage, SE, center)
    result_opened = dilate(after_erode, SE, center)
    return result_opened

# Operador morfologico: cierre
def closing(inImage, SE, center=[]):
    after_dilate = dilate(inImage, SE, center)
    result_closed = erode(after_dilate, SE, center)
    return result_closed

# Transformada hit or miss
def hit_or_miss(inImage, objSEj, bgSE, center=[]):
    if len(center) == 0:
        center = [(objSEj.shape[0] // 2), (objSEj.shape[1] // 2)]

    if objSEj.shape != bgSE.shape:
        print("Error: elementos estructurantes incoherentes")
        return None

    if (objSEj * bgSE).any():
        print("Error: elementos estructurantes incoherentes")
        return None

    after_erode_obj = erode(inImage, objSEj, center)
    inverted_input = 1 - inImage
    after_erode_bg = erode(inverted_input, bgSE, center)
    outImage = after_erode_obj * after_erode_bg

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

inImage = io.imread('morph.png')
inImage = black_and_white(inImage)
umbral = 0.5 
inImage_binary = (inImage > umbral).astype(float)

# COMPROBACION OPERADORES MORFOLOGICOS
# inImage_binary = np.array([[1, 0, 0, 0, 0, 0],
#               [1, 0, 0, 0, 0, 0],
#               [0, 1, 1, 0, 0, 0],
#               [0, 1, 0, 0, 0, 0],
#               [0, 1, 0, 0, 0, 0]], dtype=np.float32)
# SE = np.array([[1, 1]], dtype=np.float32)

# SE = np.array([[0, 0, 0, 0, 0, 0],
#               [0, 1, 1, 1, 1, 0],
#               [0, 1, 1, 1, 1, 0],
#               [0, 1, 1, 1, 1, 0],
#               [0, 0, 0, 0, 0, 0]], dtype=np.float32)

# SE = np.array([[1, 1, 1],
#                [1, 1, 1],
#                [1, 1, 1]])
# SE = np.array([[0, 0, 1, 1, 1, 0, 0]])

# outImageErode = erode(inImage_binary, SE)
# outImageDilate = dilate(inImage_binary, SE)
# outImageOpened = opening(inImage_binary, SE)
# outImageClosed = closing(inImage_binary, SE)

# saveImage(outImageErode, 'imagenes-morf/saved_erode.jpg')
# saveImage(outImageDilate, 'imagenes-morf/saved_dilate.jpg')
# saveImage(outImageOpened, 'imagenes-morf/saved_opening.jpg')
# saveImage(outImageClosed, 'imagenes-morf/saved_closing.jpg')

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
# plt.show()

# COMPROBACION HIT-OR-MISS
objSEj = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])

bgSE = np.array([[0, 1, 0],
                [0, 0, 1],
                [0, 0, 0]])

# inImage_binary = np.array([[0, 0, 0, 0, 0, 0, 0],
#                     [0, 0, 0, 1, 1, 0, 0],
#                     [0, 0, 1, 1, 1, 1, 0],
#                     [0, 0, 1, 1, 1, 1, 0],
#                     [0, 0, 0, 1, 1, 0, 0],
#                     [0, 0, 0, 1, 0, 0, 0],
#                     [0, 0, 0, 0, 0, 0, 0]])

outImage = hit_or_miss(inImage_binary, objSEj, bgSE)
saveImage(outImage, 'imagenes-morf/saved_hit_or_miss.jpg')

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(inImage_binary, cmap='gray', vmin=0.0, vmax=1.0)
plt.title('Imagen de entrada')
plt.subplot(1, 2, 2)
plt.imshow(outImage, cmap='gray', vmin=0.0, vmax=1.0)
plt.title('Hit or miss')
plt.tight_layout()
plt.show()