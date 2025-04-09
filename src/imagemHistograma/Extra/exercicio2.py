import cv2
import numpy as np
import matplotlib.pyplot as plt

def equalizar_histograma_manual(img_gray):
    hist = np.bincount(img_gray.flatten(), minlength=256)
    cdf = hist.cumsum()
    cdf_normalizada = cdf / cdf[-1]
    equalizacao = np.floor(255 * cdf_normalizada).astype(np.uint8)
    img_equalizada = equalizacao[img_gray]
    return img_equalizada, hist, cdf

# Carregando a imagem em tons de cinza
image = cv2.imread("data/paisagemPinguins.jpg", cv2.IMREAD_GRAYSCALE)

# Equalização do histograma
eqImageFuncao, _, _ = equalizar_histograma_manual(image)
eqImageEqHist = cv2.equalizeHist(image)

# Cálculo dos histogramas
histImage = cv2.calcHist([image], [0], None, [256], [0, 256])
histImageFuncao = cv2.calcHist([eqImageFuncao], [0], None, [256], [0, 256])
histImageEqHist = cv2.calcHist([eqImageEqHist], [0], None, [256], [0, 256])

# Mostrar imagens
def mostrar_imagens(titulo, original, equalizadaFuncao, equalizadaEqHist):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.title(f'{titulo} - Original')
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title(f'{titulo} - Equalizada (Função)')
    plt.imshow(equalizadaFuncao, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.title(f'{titulo} - Equalizada (OpenCV)')
    plt.imshow(equalizadaEqHist, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Mostrar histogramas
def mostrar_histograma(titulo, hist_original, hist_eq_funcao, hist_eq_eqHist):
    plt.figure(figsize=(15, 4))
    plt.subplot(1, 3, 1)
    plt.title(f'{titulo} - Original')
    plt.plot(hist_original, color='black')
    plt.xlim([0, 256])
    
    plt.subplot(1, 3, 2)
    plt.title(f'{titulo} - Equalizado (Função)')
    plt.plot(hist_eq_funcao, color='black')
    plt.xlim([0, 256])

    plt.subplot(1, 3, 3)
    plt.title(f'{titulo} - Equalizado (OpenCV)')
    plt.plot(hist_eq_eqHist, color='black')
    plt.xlim([0, 256])
    
    plt.tight_layout()
    plt.show()

# Exibir os resultados
mostrar_imagens("Imagem de Baixo Contraste", image, eqImageFuncao, eqImageEqHist)
mostrar_histograma("Imagem de Baixo Contraste", histImage, histImageFuncao, histImageEqHist)
