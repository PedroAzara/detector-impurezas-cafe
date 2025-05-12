# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:21:08 2025

@author: ramoe
"""

import os
import glob
import shutil
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import tifffile # Para garantir leitura de TIFs complexos, se necessário
import logging

# --- Configurações ---
SOURCE_IMAGE_DIR = 'tif' # !!! MUDE ISSO !!!
DATASET_BASE_DIR = './cafe_milho_dataset' # Onde o dataset formatado será criado
TRAIN_RATIO = 0.7
IMAGE_FORMAT_OUT = 'png' # Formato para conversão (png ou jpg)
MODEL_TO_USE = 'yolov8n-cls.pt' # Modelo inicial (pequeno e rápido)
NUM_EPOCHS = 50 # Número de épocas de treinamento (ajuste conforme necessário)
IMAGE_SIZE = 128 # Tamanho da imagem para treino (ajuste conforme necessário)

CLASSES = ['Cafe', 'Milho']

# Configurar logging para ver o que está acontecendo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Preparação dos Dados ---

def get_label_from_filename(filename):
    """Extrai o rótulo ('Cafe' ou 'Milho') do nome do arquivo."""
    basename = os.path.basename(filename).lower()
    if 'cafeara' in basename: # Assumindo que 'cafeara' significa café puro
        return 'Cafe'
    elif 'milho' in basename:
        return 'Milho'
    else:
        logging.warning(f"Não foi possível determinar o rótulo para: {filename}")
        return None

def prepare_datasets(source_dir, target_base_dir, train_ratio, classes, img_format_out):
    """Encontra imagens, divide, converte e organiza nos diretórios do dataset."""
    logging.info(f"Procurando arquivos .tif em: {source_dir}")
    tif_files = glob.glob(os.path.join(source_dir, '*.tif')) + glob.glob(os.path.join(source_dir, '*.tiff'))
    logging.info(f"Encontrados {len(tif_files)} arquivos TIF.")

    if not tif_files:
        logging.error("Nenhum arquivo TIF encontrado no diretório de origem. Verifique o caminho.")
        return False

    image_paths = []
    labels = []

    for f in tif_files:
        label = get_label_from_filename(f)
        if label:
            image_paths.append(f)
            labels.append(label)
        else:
             logging.warning(f"Ignorando arquivo sem rótulo claro: {f}")

    if not image_paths:
        logging.error("Nenhuma imagem com rótulo válido ('Cafe' ou 'Milho') encontrada.")
        return False

    logging.info(f"Total de imagens com rótulos válidos: {len(image_paths)}")
    logging.info(f"Distribuição inicial - Cafe: {labels.count('Cafe')}, Milho: {labels.count('Milho')}")

    # Dividir em treino e validação estratificadamente
    try:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels,
            train_size=train_ratio,
            random_state=42, # Para reprodutibilidade
            stratify=labels # Garante proporção das classes nos dois conjuntos
        )
        logging.info(f"Divisão - Treino: {len(train_paths)} imagens, Validação: {len(val_paths)} imagens")
        logging.info(f"Distribuição Treino - Cafe: {train_labels.count('Cafe')}, Milho: {train_labels.count('Milho')}")
        logging.info(f"Distribuição Validação - Cafe: {val_labels.count('Cafe')}, Milho: {val_labels.count('Milho')}")

    except ValueError as e:
         logging.error(f"Erro ao dividir os dados: {e}. Verifique se há amostras suficientes de cada classe.")
         return False

    # Criar estrutura de diretórios
    train_dir = os.path.join(target_base_dir, 'train')
    val_dir = os.path.join(target_base_dir, 'val')

    # Limpa diretórios antigos se existirem
    if os.path.exists(target_base_dir):
        logging.warning(f"Removendo diretório de dataset existente: {target_base_dir}")
        shutil.rmtree(target_base_dir)

    # Cria as pastas das classes dentro de train e val
    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Função para converter e copiar
    def convert_and_copy(paths, labels, target_set_dir):
        count = 0
        skipped_conversion = 0
        for img_path, label in zip(paths, labels):
            try:
                # Tenta abrir com Pillow primeiro, fallback para tifffile
                try:
                    img = Image.open(img_path)
                except Exception as pil_error:
                    logging.warning(f"Pillow não conseguiu abrir {img_path} ({pil_error}), tentando tifffile...")
                    # Ler com tifffile e converter para numpy array
                    img_array = tifffile.imread(img_path)
                    # Lidar com diferentes shapes e tipos de dados (simplificação comum: pegar 1o canal se multi-canal, converter para 8bit)
                    if img_array.ndim == 3:
                        # Se for multicanal (altura, largura, canais), tenta converter para RGB ou pega o primeiro canal
                        if img_array.shape[2] >= 3:
                           img_array = img_array[:, :, :3] # Pega os 3 primeiros canais (assume RGB-like)
                        else:
                            img_array = img_array[:,:,0] # Pega o primeiro canal
                    elif img_array.ndim > 3: # Ex: (frames, altura, largura, canais)
                        logging.warning(f"Imagem TIF com muitas dimensões: {img_path}, pegando o primeiro frame/slice.")
                        img_array = img_array[0]
                        if img_array.ndim == 3: # Verifica novamente se o frame é multicanal
                            if img_array.shape[2] >= 3:
                                img_array = img_array[:, :, :3]
                            else:
                                img_array = img_array[:,:,0]

                    # Normalizar para 0-255 (8-bit) se necessário (assumindo dados não-flutuantes)
                    if img_array.dtype != np.uint8:
                        if np.issubdtype(img_array.dtype, np.floating):
                             img_array = (img_array - np.min(img_array)) / (np.max(img_array) - np.min(img_array) + 1e-6) * 255
                        elif np.max(img_array) > 255: # Ex: 16-bit
                            img_array = (img_array / np.max(img_array) * 255)

                        img_array = img_array.astype(np.uint8)


                    img = Image.fromarray(img_array) # Converte array numpy para imagem Pillow

                # Converter para RGB para garantir compatibilidade (YOLO espera 3 canais)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Salvar no formato desejado
                base_filename = os.path.basename(img_path)
                new_filename = os.path.splitext(base_filename)[0] + '.' + img_format_out
                target_path = os.path.join(target_set_dir, label, new_filename)
                img.save(target_path)
                count += 1
            except Exception as e:
                logging.error(f"Falha ao converter ou salvar {img_path}: {e}")
                skipped_conversion += 1
        logging.info(f"Convertidas e copiadas {count} imagens para {target_set_dir}. Falhas/puladas: {skipped_conversion}")

    # Processar treino e validação
    logging.info("Processando conjunto de Treinamento...")
    convert_and_copy(train_paths, train_labels, train_dir)
    logging.info("Processando conjunto de Validação...")
    convert_and_copy(val_paths, val_labels, val_dir)

    logging.info("Preparação do dataset concluída.")
    return True

# --- 2. Treinamento ---

def train_yolo_classifier(dataset_dir, model_name, epochs, img_size):
    """Treina um modelo de classificação YOLOv8."""
    logging.info(f"Iniciando o treinamento com o modelo: {model_name}")
    logging.info(f"Dataset: {dataset_dir}")
    logging.info(f"Épocas: {epochs}, Tamanho da imagem: {img_size}")

    try:
        # Carrega o modelo de classificação pré-treinado
        model = YOLO(model_name)

        # Treina o modelo
        results = model.train(
            data=dataset_dir, # Diretório pai contendo 'train' e 'val'
            epochs=epochs,
            imgsz=img_size,
            batch=-1, # -1 para auto-batch (YOLO decide baseado na memória GPU) ou defina um número (ex: 16)
            name='cafe_milho_classifier_run' # Nome do experimento (será salvo em runs/classify/)
        )
        logging.info("Treinamento concluído!")
        logging.info(f"Resultados salvos em: {results.save_dir}") # results é um objeto, não um dict simples na v8
        return results.save_dir # Retorna o caminho onde os resultados foram salvos

    except Exception as e:
        logging.error(f"Erro durante o treinamento YOLO: {e}")
        import traceback
        traceback.print_exc() # Imprime mais detalhes do erro
        return None

# --- Execução ---

if __name__ == "__main__":
    # Garante que o diretório de origem existe
    if not os.path.isdir(SOURCE_IMAGE_DIR):
         logging.error(f"Diretório de origem das imagens não encontrado: {SOURCE_IMAGE_DIR}")
         logging.error("Por favor, edite a variável SOURCE_IMAGE_DIR no script.")
    else:
        # 1. Preparar os dados
        dataset_ready = prepare_datasets(SOURCE_IMAGE_DIR, DATASET_BASE_DIR, TRAIN_RATIO, CLASSES, IMAGE_FORMAT_OUT)

        if dataset_ready:
            # 2. Treinar o modelo
            results_path = train_yolo_classifier(DATASET_BASE_DIR, MODEL_TO_USE, NUM_EPOCHS, IMAGE_SIZE)

            if results_path:
                logging.info(f"\n--- Análise Pós-Treino ---")
                logging.info(f"O modelo treinado e os resultados estão em: {results_path}")
                logging.info("Verifique os arquivos gerados lá, como:")
                logging.info(f"- results.png / results.csv: Gráficos e dados de desempenho.")
                logging.info(f"- confusion_matrix.png: Matriz de confusão (importante para ver erros entre Cafe/Milho).")
                logging.info(f"- weights/best.pt: O melhor modelo treinado.")

                logging.info("\nPara tentar entender as 'diferenças' que o modelo aprendeu:")
                logging.info("1. Analise a 'confusion_matrix.png': Veja se o modelo confunde mais 'Cafe' com 'Milho' ou vice-versa.")
                logging.info("2. Olhe as imagens no conjunto de validação ({}/val/...) que foram classificadas INCORRETAMENTE.")
                logging.info("   - O que há de visualmente diferente ou semelhante nessas imagens que pode ter confundido o modelo?")
                logging.info("   - Há grãos de milho muito sutis? A iluminação é diferente? A textura é ambígua?")
                logging.info("3. (Avançado) Pesquise por 'Grad-CAM YOLOv8 classification' para encontrar tutoriais sobre como gerar mapas de calor")
                logging.info("   que mostram quais pixels foram mais importantes para a decisão do modelo em imagens específicas.")