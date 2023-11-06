import numpy as np
from skimage import io
from skimage import color
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def adjustIntensity(inImage, inRange=None, outRange=[0, 1]):
    if inRange is None:
        inRange = [np.min(inImage), np.max(inImage)]
    
    imin, imax = inRange
    omin, omax = outRange
    
    outImage = (((inImage - imin) / (imax - imin)) * (omax - omin)) + omin

    outImage[outImage < imin] = imin
    outImage[outImage > imax] = imax

    return outImage

def equalizeIntensity(inImage, nBins=256):
    hist, bin_edges = np.histogram(inImage, bins=nBins, range=(0, 1))

    cdf = hist.cumsum()

    cdf_normalized = cdf / cdf[-1]

    interp_func = interp1d(bin_edges[:-1], cdf_normalized, kind='linear', fill_value='extrapolate')

    outImage = interp_func(inImage)

    return outImage

def saveImage(image, filename):
    scaled_image = (image * 255).astype(np.uint8)
    io.imsave(filename, scaled_image)

inImage = io.imread('imagenes-hist/perro.jpg')

def black_and_white(img):
    if len(img.shape) == 2:
        img = img.astype(float) / 255.0
    elif len(img.shape) == 3:
        if img.shape[2] == 4:  
            img = img[:, :, :3] 
        img = color.rgb2gray(img).astype(float)
    return img

inImageBW = black_and_white(inImage)

# outImage = adjustIntensity(inImageBW, inRange=[0,1], outRange=[0, 0.5])
outImage = equalizeIntensity(inImageBW, nBins=256)

saveImage(outImage, 'imagenes-hist/imagen_guardada_equalizeIntensity.jpg')

# Histograma de la imagen original
plt.subplot(1, 2, 1)
plt.hist(inImageBW.ravel(), bins=256, range=(0, 1), color='blue', alpha=0.7)
plt.title('Histograma original')

# Histograma de la imagen ajustada
plt.subplot(1, 2, 2)
plt.hist(outImage.ravel(), bins=256, range=(0, 1), color='red', alpha=0.7)
plt.title('Histograma ajustado')

# Imágenes
plt.figure()
plt.subplot(1, 2, 1)
io.imshow(inImageBW, cmap='gray') 
plt.title('Imagen original')
plt.subplot(1, 2, 2)
io.imshow(outImage, cmap='gray')
plt.title('Imagen ajustada')

plt.show()

