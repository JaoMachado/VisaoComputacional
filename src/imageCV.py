import pandas as pd
import numpy as np
import cv2
import random

imageOriginal = cv2.imread("data/macau-parrot.jpg")
image = cv2.imread("data/macau-parrot.jpg")

dimensoes = image.shape
num_elementos = image.size
tipo_imagem = image.dtype

print(f"Dimensão: {dimensoes}")
print(f"Tamanho.: {num_elementos}")
print(f"Tipo....: {tipo_imagem}")

def criaRandom():
    prim = random.randint(1, 2997)
    seg = random.randint(1, 1999)
    pixel = [prim, seg]
    return pixel

def criarRgb():
    rgb = []
    for a in range(3):
        b = random.randint(0, 255)
        rgb.append(b)
    return rgb
lista = []
for a in range(1000000):
    lista.append(criaRandom())

for linha, coluna in lista:
    image[linha][coluna] = criarRgb()

cv2.imshow("Imagem", imageOriginal)
cv2.waitKey(0) 
cv2.imshow("Imagem", image)
cv2.waitKey(0) 