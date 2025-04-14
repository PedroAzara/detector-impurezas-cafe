import cv2
import numpy as np
import os
import pandas as pd
from skimage.measure import shannon_entropy
from skimage.feature import graycomatrix, graycoprops

# Digite aqui qual tonelada de café você quer rodar o script
t = 7  # Exemplo: 1, 4, 7
# Caminho da pasta com as imagens
pasta = rf"C:\Users\pedro\OneDrive\Documentos\projetos\imagens\imagens-separadas\{t}T\02-04-25"

dados = []

for nome_arquivo in os.listdir(pasta):
    if nome_arquivo.endswith((".png", ".jpg", ".jpeg")):
        caminho = os.path.join(pasta, nome_arquivo)
        imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)

        intensidade = np.mean(imagem)
        desvio = np.std(imagem)
        entropia = shannon_entropy(imagem)

        # Reduz a imagem se for muito grande para o GLCM
        if imagem.shape[0] > 512 or imagem.shape[1] > 512:
            imagem_resized = cv2.resize(imagem, (256, 256))
        else:
            imagem_resized = imagem

        # Matriz de coocorrência (GLCM)
        glcm = graycomatrix(imagem_resized, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contraste = graycoprops(glcm, 'contrast')[0, 0]
        homogeneidade = graycoprops(glcm, 'homogeneity')[0, 0]

        # Identifica a classe
        if "cafe_Luisa" in nome_arquivo:
            classe = "cafe_Luisa"
        elif "cafe_" in nome_arquivo.lower():
            classe = "cafe_padrao"
        elif "milho1" in nome_arquivo.lower():
            classe = "milho1"
        elif "milho3" in nome_arquivo.lower():
            classe = "milho3"
        else:
            classe = "desconhecida"

        dados.append({
            "arquivo": nome_arquivo,
            "classe": classe,
            "intensidade_media": intensidade,
            "desvio_padrao": desvio,
            "entropia": entropia,
            "contraste": contraste,
            "homogeneidade": homogeneidade
        })

# Criar DataFrame e salvar como CSV
df = pd.DataFrame(dados)

pasta_resultados = r"C:\Users\pedro\OneDrive\Documentos\projetos\resultados"
os.makedirs(pasta_resultados, exist_ok=True)

caminho_csv = os.path.join(pasta_resultados, f"metricas_textura_{t}T.csv")
df.to_csv(caminho_csv, index=False)
print(f"CSV com métricas salvo em: {caminho_csv}")
