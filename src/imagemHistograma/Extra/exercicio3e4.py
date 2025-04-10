import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregando a imagem em tons de cinza
image = cv2.imread("data/deckPB.jpg", cv2.IMREAD_GRAYSCALE)

# 1. Negativo da imagem
negativo = 255 - image

# 2. Transformação logarítmica
imagem_log = np.log1p(image.astype(np.float32))  # log(1 + I)
imagem_log = cv2.normalize(imagem_log, None, 0, 255, cv2.NORM_MINMAX)
imagem_log = imagem_log.astype(np.uint8)

# 3. Transformações de potência (correção gama)
def transformacao_potencia(img, gamma):
    img_float = img.astype(np.float32) / 255.0
    img_corrigida = np.power(img_float, gamma)
    img_corrigida = np.uint8(img_corrigida * 255)
    return img_corrigida

gamma05 = transformacao_potencia(image, 0.5)
gamma1 = transformacao_potencia(image, 1.0)
gamma2 = transformacao_potencia(image, 2.0)

# Cálculo dos histogramas
histImage = cv2.calcHist([image], [0], None, [256], [0, 256])
histNegativo = cv2.calcHist([negativo], [0], None, [256], [0, 256])
histLog = cv2.calcHist([imagem_log], [0], None, [256], [0, 256])
histGamma05 = cv2.calcHist([gamma05], [0], None, [256], [0, 256])
histGamma1 = cv2.calcHist([gamma1], [0], None, [256], [0, 256])
histGamma2 = cv2.calcHist([gamma2], [0], None, [256], [0, 256])

# Mostrar imagens
def mostrar_imagens(titulo, original, negativo, logaritmica, gamma05, gamma1, gamma2):
    plt.figure(figsize=(18, 8))
    plt.subplot(2, 3, 1)
    plt.title(f'{titulo} - Original')
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title(f'{titulo} - Negativo')
    plt.imshow(negativo, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title(f'{titulo} - Logaritmica')
    plt.imshow(logaritmica, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title(f'{titulo} - Gamma 0.5')
    plt.imshow(gamma05, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title(f'{titulo} - Gamma 1.0')
    plt.imshow(gamma1, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title(f'{titulo} - Gamma 2.0')
    plt.imshow(gamma2, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Mostrar histogramas
def mostrar_histograma(titulo, hist_original, histNegativo, histLog, histGamma05, histGamma1, histGamma2):
    plt.figure(figsize=(18, 8))
    plt.subplot(2, 3, 1)
    plt.title(f'{titulo} - Original')
    plt.plot(hist_original, color='black')
    plt.xlim([0, 256])
    
    plt.subplot(2, 3, 2)
    plt.title(f'{titulo} - Negativo')
    plt.plot(histNegativo, color='black')
    plt.xlim([0, 256])

    plt.subplot(2, 3, 3)
    plt.title(f'{titulo} - Logaritmico')
    plt.plot(histLog, color='black')
    plt.xlim([0, 256])

    plt.subplot(2, 3, 4)
    plt.title(f'{titulo} - Gamma 0.5')
    plt.plot(histGamma05, color='black')
    plt.xlim([0, 256])

    plt.subplot(2, 3, 5)
    plt.title(f'{titulo} - Gamma 1.0')
    plt.plot(histGamma1, color='black')
    plt.xlim([0, 256])

    plt.subplot(2, 3, 6)
    plt.title(f'{titulo} - Gamma 2.0')
    plt.plot(histGamma2, color='black')
    plt.xlim([0, 256])
    
    plt.tight_layout()
    plt.show()

# Exibir os resultados
mostrar_imagens("Imagem", image, negativo, imagem_log, gamma05, gamma1, gamma2)
mostrar_histograma("Histograma", histImage, histNegativo, histLog, histGamma05, histGamma1, histGamma2)
