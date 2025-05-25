import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Digite aqui qual tonelada de café você quer rodar o script
t = 7
# Caminho da pasta com as imagens
pasta = rf"C:\Users\pedro\OneDrive\Documentos\projetos\imagens\imagens-separadas\{t}T\02-04-25"

# Dicionário para armazenar intensidades
intensidades = {"cafe_padrao": [], "cafe_Luisa": [],"milho1":[], "milho3": []}

# Loop nas imagens
for nome_arquivo in os.listdir(pasta):
    if nome_arquivo.endswith((".png", ".jpg", ".jpeg")):
        caminho = os.path.join(pasta, nome_arquivo)
        imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        intensidade = np.mean(imagem)

        # Adiciona a intensidade à lista correspondente
        if "cafe_Luisa" in nome_arquivo:
            intensidades["cafe_Luisa"].append(intensidade)
        elif "cafe_" in nome_arquivo.lower():
            intensidades["cafe_padrao"].append(intensidade)
        elif "milho1" in nome_arquivo.lower():
            intensidades["milho1"].append(intensidade)
        elif "milho3" in nome_arquivo.lower():
            intensidades["milho3"].append(intensidade)


# Plotar gráfico de comparação
plt.figure(figsize=(10, 5))
plt.plot(intensidades["cafe_padrao"], 'o-', label='Café Padrão')
plt.plot(intensidades["milho3"], 's-', label='Milho 3')
plt.plot(intensidades["cafe_Luisa"], 'o-', label='Café Luisa')
plt.plot(intensidades["milho1"], 's-', label='Milho 1')
plt.title("Intensidade média das imagens 02-04-25")
plt.xlabel("Amostra")
plt.ylabel("Intensidade média (0-255)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# Criar pasta "resultados" se não existir
pasta_resultados = r"C:\Users\pedro\OneDrive\Documentos\projetos\resultados"
os.makedirs(pasta_resultados, exist_ok=True)

# Salvar o gráfico na pasta "resultados"
caminho_grafico = os.path.join(pasta_resultados, f"intensidade_comparacao_02-04-25_{t}T.png")
plt.savefig(caminho_grafico)
print(f"Gráfico salvo em: {caminho_grafico}")
plt.show()
