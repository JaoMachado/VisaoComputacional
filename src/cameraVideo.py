import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Inicializa a captura de vídeo (0 para webcam padrão)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Erro ao abrir camera")
    exit()

while True:
    ret, frame = cap.read()  # Lê um frame da câmera
    if not ret:
        break

    cv2.imshow("Camera", frame)  # Exibe o frame

    # Pressione 'esc' para sair
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()  # Libera a câmera
cv2.destroyAllWindows()

video_path = "data/video.mp4"
video = cv2.VideoCapture(video_path)

while video.isOpened():
    ret, frame = video.read()
    if not ret:
        break  # Sai do loop quando o vídeo terminar

    cv2.imshow("Video", frame)  # Exibe o frame

    # Pressione 'esc' para sair
    if cv2.waitKey(25) & 0xFF == 27:
        break

video.release()  # Libera o vídeo
cv2.destroyAllWindows()

image = cv2.imread("data/macau-parrot.jpg")

# Desenhar um retângulo (imagem, ponto inicial, ponto final, cor, espessura)
cv2.rectangle(image, (1073, 890), (1670, 1570), (0, 255, 0), 3)

# Desenhar um círculo (imagem, centro, raio, cor, espessura)
cv2.circle(image, (1110, 2800), 50, (255, 0, 0), -1)  # Preenchido

# Desenhar uma linha (imagem, ponto inicial, ponto final, cor, espessura)
cv2.line(image, (730, 130), (1375, 130), (0, 0, 0), 3)

# Adicionar um texto (imagem, texto, posição, fonte, escala, cor, espessura)
cv2.putText(image, "Macau Parrot", (730, 118), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 3)

# Mostrar a imagem
cv2.imshow("Desenho OpenCV", image)
cv2.waitKey(0)
cv2.destroyAllWindows()