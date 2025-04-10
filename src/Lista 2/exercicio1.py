import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Lendo as duas imagens
image1 = cv2.imread("data/JGL.png")
image2 = cv2.imread("data/macau-parrot.jpg")

# Pegar as dimensões da primeira imagem
height, width = image1.shape[:2]

# Redimensionar a segunda imagem para ter o mesmo tamanho da primeira
image2_resized = cv2.resize(image2, (width, height))

# Soma com Clipping (cv2.add faz isso automaticamente)
soma_clipping = cv2.add(image1, image2_resized)

# Soma com Média (para evitar estouro: (img1 + img2) / 2)
soma_media = cv2.addWeighted(image1, 0.5, image2_resized, 0.5, 0 )

# Função para mostrar imagens lado a lado
def mostrar_imagens(titulo, clipping, media):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f'{titulo} - Clipping')
    plt.imshow(clipping, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'{titulo} - Media')
    plt.imshow(media, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

mostrar_imagens("Imagem", soma_clipping, soma_media)