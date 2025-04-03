import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

image = cv2.imread("data/deckPB.jpg")

histograma = cv2.calcHist([image], [0], None, [256], [0, 256])

plt.figure()
plt.title("Histograma da Imagem")
plt.xlabel("Intensidade de Pixel")
plt.ylabel("Frequência")
plt.plot(histograma, color='black')  # Exibir histograma em preto
plt.xlim([0, 256])  # Definir o intervalo do eixo X
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
