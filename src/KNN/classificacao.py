import cv2 
import numpy as np
from sklearn.datasets import load_digits

# Carregar o dataset de digitos
digitos = load_digits()

# Preparando os dados
imagens = digitos.images
rotulos = digitos.target

# Selecionar imagens com rotulos 0 e 1
mascara = (rotulos == 0) | (rotulos == 1)

# Filtrando
imagens = imagens[mascara]
rotulos = rotulos[mascara]

# Extrair as características
features = []

for im in imagens:
    media = np.mean(im) # media dos pixels
    desvio = np.std(im) # desvio padrao dos pixels
    pixels = np.sum(im >= 8)
    features.append([media, desvio, pixels])

# Converter a lista de caracteristicas em um array numpy
features = np.array(features, dtype=np.float32)
rotulos = rotulos.astype(int)

# Dividir os dados em conjuntos de treinamento e teste
taxa_treino = 0.8

# Embaralhar os dados
id = np.arange(len(rotulos))
np.random.shuffle(id) # para misturar os ids
features = features[id]
rotulos = rotulos[id]

split = int(taxa_treino * len(rotulos)) # calculo de onde dividir
dados_treino, dados_teste = features[:split], features[split:] # dividindo os dados para treino e teste ( 80% treino e 20% teste)
rotulos_treino, rotulos_teste = rotulos[:split], rotulos[split:] 

# Criar o classificador/modelo
knn = cv2.ml.KNearest_create()

# Treinar o classificador
knn.train(dados_treino, cv2.ml.ROW_SAMPLE, rotulos_treino)

# Definir o k
k = 3

# Encontrar os vizinhos mais proximos
ret, rotulos_classificados, vizinhos, distancias = knn.findNearest(dados_teste, k)

print(rotulos_classificados)