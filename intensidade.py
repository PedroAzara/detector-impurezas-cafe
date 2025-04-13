import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Caminho da pasta com as imagens
pasta = r"C:\Users\pedro\OneDrive\Documentos\projetos\imagens\imagens-separadas\1T\01-04-25"

# Dicionário para armazenar intensidades
intensidades = {"cafe": [], "milho": []}

# Loop nas imagens
for nome_arquivo in os.listdir(pasta):
    if nome_arquivo.endswith((".png", ".jpg", ".jpeg")):
        caminho = os.path.join(pasta, nome_arquivo)
        imagem = cv2.imread(caminho, cv2.IMREAD_GRAYSCALE)
        intensidade = np.mean(imagem)

        if "cafe" in nome_arquivo.lower():
            intensidades["cafe"].append(intensidade)
        elif "milho" in nome_arquivo.lower():
            intensidades["milho"].append(intensidade)

# Exibir resultados
print("Intensidade média - Café:", intensidades["cafe"])
print("Intensidade média - Milho:", intensidades["milho"])

# Plotar gráfico de comparação
plt.figure(figsize=(10, 5))
plt.plot(intensidades["cafe"], 'o-', label='Café')
plt.plot(intensidades["milho"], 's-', label='Milho')
plt.title("Intensidade média das imagens 01-04-25")
plt.xlabel("Amostra")
plt.ylabel("Intensidade média (0-255)")
plt.legend()
plt.grid(True)
plt.tight_layout()
# Criar pasta "resultados" se não existir
pasta_resultados = r"C:\Users\pedro\OneDrive\Documentos\projetos\resultados"
os.makedirs(pasta_resultados, exist_ok=True)

# Salvar o gráfico na pasta "resultados"
caminho_grafico = os.path.join(pasta_resultados, "intensidade_comparacao_01-04-25.png")
plt.savefig(caminho_grafico)
print(f"Gráfico salvo em: {caminho_grafico}")
plt.show()
