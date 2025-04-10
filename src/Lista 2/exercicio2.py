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

# Converter para float para multiplicação
img1_float = image1.astype(np.float32)
img2_float = image2_resized.astype(np.float32)

# Subtracao com Clipping
sub_clipping = np.clip(image1.astype(np.int16) - image2_resized.astype(np.int16), 0, 255).astype(np.uint8)

# Multiplicacao com Clipping
mult_clipping = img1_float * img2_float / 255.0
mult_clipping = np.clip(mult_clipping, 0, 255).astype(np.uint8)

# Divisao com Clipping
div_safe = img2_float.copy()
div_safe[div_safe == 0] = 1 
div_clipping = img1_float / div_safe * 255.0
div_clipping = np.clip(div_clipping, 0, 255).astype(np.uint8)

# Subtracao com Média (para evitar estouro: (img1 - img2) / 2)
mediaSub = ((image1.astype(np.float32) + image2_resized.astype(np.float32)) / 2).astype(np.uint8)
sub_media = cv2.subtract(image1, mediaSub)

# Multiplicacao com Média
media = ((img1_float + img2_float) / 2.0)
mult_media = (img1_float * img2_float) / 255.0
mult_media = ((mult_media + media) / 2.0)
mult_media = np.clip(mult_media, 0, 255).astype(np.uint8)

# Divisao com Media
div_media = (img1_float / div_safe * 255.0)
div_media = ((div_media + media) / 2.0)
div_media = np.clip(div_media, 0, 255).astype(np.uint8)

# Função para mostrar imagens lado a lado
def mostrar_imagens(titulo, subClipping, subMedia, multClipping, multMedia, divClipping, divMedia):
    plt.figure(figsize=(18, 8))
    plt.subplot(2, 3, 1)
    plt.title(f'{titulo} - Subtracao com Clipping')
    plt.imshow(subClipping, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title(f'{titulo} - Subtracao com Media')
    plt.imshow(subMedia, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title(f'{titulo} - Multiplicacao com Clipping')
    plt.imshow(multClipping, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.title(f'{titulo} - Multiplicacao com Media')
    plt.imshow(multMedia, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.title(f'{titulo} - Divisao com Clipping')
    plt.imshow(divClipping, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title(f'{titulo} - Divisao com Media')
    plt.imshow(divMedia, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

mostrar_imagens("Imagem", sub_clipping, sub_media, mult_clipping, mult_media, div_clipping, div_media)