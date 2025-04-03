import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Função para calcular e exibir histograma
def plot_histogram(image, title, subplot):
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.subplot(2, 2, subplot)
    plt.plot(hist, color='black')
    plt.xlim([0, 256])
    plt.title(title)

imageUniforme = cv2.imread("data/deckPB.jpg", cv2.IMREAD_GRAYSCALE)

imageBaixoContraste = cv2.imread("data/paisagemPinguins.jpg")

# Equalizar histograma das imagens
imageUniformeEq = cv2.equalizeHist(imageUniforme)
imageBaixoContrasteEq = cv2.equalizeHist(imageBaixoContraste)

cv2.imshow("Original - Baixo Contraste", imageBaixoContraste)
cv2.imshow("Equalizado - Baixo Contraste", imageBaixoContrasteEq)
cv2.imshow("Original - Iluminação Não Uniforme", imageUniforme)
cv2.imshow("Equalizado - Iluminação Não Uniforme", imageUniformeEq)

# Plotar histogramas
plt.figure(figsize=(12, 6))

# Histograma original - Baixo Contraste
plot_histogram(imageBaixoContraste, "Histograma Original - Baixo Contraste", 1)
# Histograma equalizado - Baixo Contraste
plot_histogram(imageBaixoContrasteEq, "Histograma Equalizado - Baixo Contraste", 2)
# Histograma original - Iluminação Não Uniforme
plot_histogram(imageUniforme, "Histograma Original - Iluminação Não Uniforme", 3)
# Histograma equalizado - Iluminação Não Uniforme
plot_histogram(imageUniformeEq, "Histograma Equalizado - Iluminação Não Uniforme", 4)

# Mostrar gráficos
plt.tight_layout()
plt.show()

# Aguardar tecla para fechar janelas
cv2.waitKey(0)
cv2.destroyAllWindows()