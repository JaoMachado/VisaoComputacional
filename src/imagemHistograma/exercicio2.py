import cv2
import numpy as np
import matplotlib.pyplot as plt

# Carregando as imagens em tons de cinza
imageUniforme = cv2.imread("data/deckPB.jpg", cv2.IMREAD_GRAYSCALE)
imageBaixoContraste = cv2.imread("data/paisagemPinguins.jpg", cv2.IMREAD_GRAYSCALE)

# Equalização do histograma
eqImageUniforme = cv2.equalizeHist(imageUniforme)
eqImageBaixoContraste = cv2.equalizeHist(imageBaixoContraste)

# Cálculo dos histogramas
histUniforme = cv2.calcHist([imageUniforme], [0], None, [256], [0, 256])
histBaixoContraste = cv2.calcHist([imageBaixoContraste], [0], None, [256], [0, 256])
histUniformeEq = cv2.calcHist([eqImageUniforme], [0], None, [256], [0, 256])
histBaixoContrasteEq = cv2.calcHist([eqImageBaixoContraste], [0], None, [256], [0, 256])

# Função para mostrar imagens lado a lado
def mostrar_imagens(titulo, original, equalizada):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.title(f'{titulo} - Original')
    plt.imshow(original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f'{titulo} - Equalizada')
    plt.imshow(equalizada, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Função para mostrar histograma
def mostrar_histograma(titulo, hist_original, hist_eq):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.title(f'{titulo} - Histograma Original')
    plt.plot(hist_original, color='black')
    plt.xlim([0, 256])
    
    plt.subplot(1, 2, 2)
    plt.title(f'{titulo} - Histograma Equalizado')
    plt.plot(hist_eq, color='black')
    plt.xlim([0, 256])
    
    plt.tight_layout()
    plt.show()

# Mostrar resultados
mostrar_imagens("Imagem de Baixo Contraste", imageBaixoContraste, eqImageBaixoContraste)
mostrar_histograma("Imagem de Baixo Contraste", histBaixoContraste, histBaixoContrasteEq)

mostrar_imagens("Imagem com Iluminação Não Uniforme", imageUniforme, eqImageUniforme)
mostrar_histograma("Imagem com Iluminação Não Uniforme", histUniforme, histUniformeEq)
