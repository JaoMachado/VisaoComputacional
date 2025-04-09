import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregando a imagem em tons de cinza
image = cv2.imread("data/deckPB.jpg", cv2.IMREAD_GRAYSCALE)

# Calculando o Histograma
histograma = cv2.calcHist([image], [0], None, [256], [0, 256])

limiar = 100

def limiarizar_imagem(imagem, limiar):
    _, imagem_limiarizada = cv2.threshold(imagem, limiar, 255, cv2.THRESH_BINARY)
    return imagem_limiarizada

imagem_limiarizada = limiarizar_imagem(image, limiar)

histograma_limiarizada = cv2.calcHist([imagem_limiarizada], [0], None, [256], [0, 256])

# Função para mostrar imagens lado a lado
def mostrar_imagens(titulo, original, limiarizada):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f'{titulo} - Original')
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f'{titulo} - Limiarizada')
    plt.imshow(limiarizada, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Função para mostrar histograma
def mostrar_histograma(titulo, hist_original, hist_lm):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title(f'{titulo} - Original')
    plt.plot(hist_original, color='black')
    plt.xlim([0, 256])
    
    plt.subplot(1, 2, 2)
    plt.title(f'{titulo} - Limiarizado')
    plt.plot(hist_lm, color='black')
    plt.xlim([-5, 260])
    
    plt.tight_layout()
    plt.show()

# Mostrar resultados
mostrar_imagens("Imagem", image, imagem_limiarizada)
mostrar_histograma("Histograma", histograma, histograma_limiarizada)