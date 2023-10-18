import numpy as np
from skimage import io
import matplotlib.pyplot as plt

def erode(inImage, SE, center=[]):
    if len(center) == 0:
        center = [(SE.shape[0] // 2 + 1), (SE.shape[1] // 2 + 1)]
    
    P, Q = SE.shape
    M, N = inImage.shape
    outImage = np.zeros((M, N), dtype=np.uint8)

    for i in range(M):
        for j in range(N):
            match = False
            for p in range(P):
                for q in range(Q):
                    x = i + p - center[0]
                    y = j + q - center[1]
                    if x >= 0 and x < M and y >= 0 and y < N:
                        if SE[p, q] == 1 and inImage[x, y] == 0:
                            match = False
                            break
                        match = True
                if not match:
                    break
            outImage[i, j] = 255 if match else 0

    return outImage

def dilate(inImage, SE, center=[]):
    if len(center) == 0:
        center = [(SE.shape[0] // 2), (SE.shape[1] // 2)]
    
    P, Q = SE.shape
    M, N = inImage.shape
    outImage = np.zeros((M, N), dtype=np.uint8)

    for i in range(M):
        for j in range(N):
            for p in range(P):
                for q in range(Q):
                    x = i + p - center[0]
                    y = j + q - center[1]
                    if x >= 0 and x < M and y >= 0 and y < N:
                        if SE[p, q] == 1 and inImage[x, y] == 1:
                            outImage[i, j] = 255 
                        else: 0
    return outImage

def saveImage(image, filename):
    scaled_image = (image * 255).astype(np.uint8)
    io.imsave(filename, scaled_image)

inImage = io.imread('pez-binario.jpg')

threshold = 150
binarized = 1.0 * (inImage > threshold)

binarized = np.array([[0, 0, 0, 0, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [0, 0, 1, 1, 0, 0],
              [0, 0, 0, 0, 0, 0]], dtype=np.uint8)

SE = np.array([[0, 1, 1]], dtype=np.uint8)

outImageErode = erode(binarized, SE)
outImageDilate = dilate(binarized, SE)

plt.subplot(1, 3, 1)
plt.title('Imagen de entrada')
io.imshow(binarized, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Erosión')
io.imshow(outImageErode, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Dilatación')
io.imshow(outImageDilate, cmap='gray')

plt.show()
