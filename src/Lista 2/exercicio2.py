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

# Subtracao com Clipping
sub_clipping = np.clip(image1.astype(np.int16) - image2_resized.astype(np.int16), 0, 255).astype(np.uint8)

# Multiplicacao com Clipping
sub_clipping = np.clip(image1.astype(np.int16) - image2_resized.astype(np.int16), 0, 255).astype(np.uint8)

media = ((image1.astype(np.float32) + image2_resized.astype(np.float32)) / 2).astype(np.uint8)

# Subtracao com Média (para evitar estouro: (img1 - img2) / 2)
sub_media = cv2.subtract(image1, media)

# Função para mostrar imagens lado a lado
def mostrar_imagens(titulo, subClipping, subMedia, multClipping, multMedia, divClipping, divMedia):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f'{titulo} - Subtracao com Clipping')
    plt.imshow(subClipping, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title(f'{titulo} - Subtracao com Media')
    plt.imshow(subMedia, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

mostrar_imagens("Imagem", sub_clipping, sub_media)