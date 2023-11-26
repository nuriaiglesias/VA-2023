import numpy as np
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Alteración rango dinámico
def adjustIntensity(inImage, inRange=None, outRange=[0, 1]):
    if inRange is None:
        inRange = [np.min(inImage), np.max(inImage)]
    
    imin, imax = inRange
    omin, omax = outRange
    
    # Ajuste del rango dinámico a outRange
    outImage = (((inImage - imin) / (imax - imin)) * (omax - omin)) + omin

    # Asegura rango de salida entre imin, imax
    outImage[outImage < imin] = imin
    outImage[outImage > imax] = imax

    return outImage

# Ecualización del histograma
def equalizeIntensity(inImage, nBins=256):
    # Calculo del histograma
    histogram_values, bin_divisions = np.histogram(inImage, bins=nBins, range=(0, 1))

    # Calculo histograma acumulado
    accumulated_histogram = histogram_values.cumsum()
    
    # Normalizacion para estar en rango [0,1]
    accumulated_hist_normalized = accumulated_histogram / np.max(accumulated_histogram)

    # Interpolación
    interp_algorithm = interp1d(bin_divisions[:-1], accumulated_hist_normalized, kind='linear', fill_value='extrapolate')

    outImage = interp_algorithm(inImage)

    return outImage

def saveImage(image, filename):
    scaled_image = (image * 255).astype(np.uint8)
    io.imsave(filename, scaled_image)

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

inImage = io.imread('imagenes-hist/perro.jpg')
inImageBW = black_and_white(inImage)

outImage = adjustIntensity(inImageBW, outRange=[0.2, 0.3])
# outImage = equalizeIntensity(inImageBW, nBins=256)

saveImage(outImage, 'imagenes-hist/saved_equalizeIntensity.jpg')

# Histogramas
plt.subplot(1, 2, 1)
plt.hist(inImageBW.ravel(), bins=256, range=(0, 1), color='blue', alpha=0.7)
plt.title('Histograma original')
plt.subplot(1, 2, 2)
plt.hist(outImage.ravel(), bins=256, range=(0, 1), color='red', alpha=0.7)
plt.title('Histograma ajustado')

# Imágenes
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(inImageBW, cmap='gray',vmin=0.0,vmax=1.0) 
plt.title('Imagen original')
plt.subplot(1, 2, 2)
plt.imshow(outImage, cmap='gray',vmin=0.0,vmax=1.0)
plt.title('Imagen ajustada')
plt.show()