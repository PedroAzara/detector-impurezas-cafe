import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.color import rgb2gray  # caso a imagem esteja em RGB
from skimage.measure import shannon_entropy

# Caminho da pasta com as imagens
pasta = r"C:\Users\pedro\OneDrive\Documentos\projetos\imagens\imagens-separadas\1T\02-04-25"

# Dicionário para armazenar intensidades
intensidades = {"cafe_padrao": [], "cafe_Luisa": [],"milho1":[], "milho3": []}

# Loop nas imagens
for nome_arquivo in os.listdir(pasta):
    if nome_arquivo.endswith((".png", ".jpg", ".jpeg")):
        caminho = os.path.join(pasta, nome_arquivo)
        imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        entropia = shannon_entropy(imagem)

        # Adiciona a intensidade à lista correspondente
        if "cafe_Luisa" in nome_arquivo:
            intensidades["cafe_Luisa"].append(entropia)
        elif "cafe_" in nome_arquivo.lower():
            intensidades["cafe_padrao"].append(entropia)
        elif "milho1" in nome_arquivo.lower():
            intensidades["milho1"].append(entropia)
        elif "milho3" in nome_arquivo.lower():
            intensidades["milho3"].append(entropia)


# Plotar gráfico de comparação
plt.figure(figsize=(10, 5))
plt.plot(intensidades["cafe_padrao"], 'o-', label='Café Padrão')
plt.plot(intensidades["milho3"], 's-', label='Milho 3')
plt.plot(intensidades["cafe_Luisa"], 'o-', label='Café Luisa')
plt.plot(intensidades["milho1"], 's-', label='Milho 1')
plt.title("Entropia de Shannon das imagens 02-04-25")
plt.xlabel("Amostra")
plt.ylabel("Entropia de Shannon")
plt.legend()
plt.grid(True)
plt.tight_layout()
# Criar pasta "resultados" se não existir
pasta_resultados = r"C:\Users\pedro\OneDrive\Documentos\projetos\resultados"
os.makedirs(pasta_resultados, exist_ok=True)

# Salvar o gráfico na pasta "resultados"
caminho_grafico = os.path.join(pasta_resultados, "shannon_entropy_02-04-25.png")
plt.savefig(caminho_grafico)
print(f"Gráfico salvo em: {caminho_grafico}")
plt.show()
