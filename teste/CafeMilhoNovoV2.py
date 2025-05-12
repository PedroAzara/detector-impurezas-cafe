# -*- coding: utf-8 -*-
"""
Created on Mon May 12 09:32:34 2025

@author: ramoe
"""

import os
import glob
import shutil
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import tifffile
import logging
import math
import matplotlib.pyplot as plt
import seaborn as sns
import traceback # Para imprimir detalhes do erro

# --- Configurações ---
SOURCE_IMAGE_DIR = 'tif' # !!! MUDE ISSO !!!
DATASET_BASE_DIR = './cafe_milho_dataset_classif' # Onde o dataset formatado será criado (renomeado para evitar conflito)
TRAIN_RATIO = 0.7
IMAGE_FORMAT_OUT = 'png' # Formato para conversão (png ou jpg)
MODEL_TO_USE = 'yolov8n-cls.pt' # Modelo inicial (pequeno e rápido)
NUM_EPOCHS = 50 # Número de épocas de treinamento (ajuste conforme necessário)
IMAGE_SIZE = 128 # Tamanho da imagem para treino (ajuste conforme necessário)

CLASSES = ['Cafe', 'Milho']

# Configurar logging para ver o que está acontecendo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Funções de Preparação de Dados (mantidas do código anterior) ---

def get_label_from_filename(filename):
    """Extrai o rótulo ('Cafe' ou 'Milho') do nome do arquivo."""
    basename = os.path.basename(filename).lower()
    # Ser mais específico para evitar falsos positivos se 'cafe' ou 'milho' aparecerem em outros contextos
    if basename.startswith('cafeara'):
        return 'Cafe'
    elif basename.startswith('milho'):
        return 'Milho'
    else:
        # Tentativa secundária (menos específica)
        if 'cafeara' in basename:
             logging.warning(f"Usando regra menos específica para {filename} -> Cafe")
             return 'Cafe'
        elif 'milho' in basename:
             logging.warning(f"Usando regra menos específica para {filename} -> Milho")
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
        return False, None # Retorna status e None para caminhos

    image_paths = []
    labels = []

    for f in tif_files:
        label = get_label_from_filename(f)
        if label:
            image_paths.append(f)
            labels.append(label)
        else:
             logging.warning(f"Ignorando arquivo sem rótulo claro: {f}")

    if not image_paths or len(set(labels)) < len(classes):
        logging.error(f"Não foram encontradas imagens suficientes para todas as classes ({classes}). Encontrados rótulos: {set(labels)}")
        return False, None

    logging.info(f"Total de imagens com rótulos válidos: {len(image_paths)}")
    logging.info(f"Distribuição inicial - Cafe: {labels.count('Cafe')}, Milho: {labels.count('Milho')}")

    # Garantir que há pelo menos 2 amostras por classe para estratificação
    min_samples_per_class = min(labels.count(c) for c in set(labels))
    if min_samples_per_class < 2:
        logging.error(f"Pelo menos uma classe tem menos de 2 amostras ({dict(zip(*np.unique(labels, return_counts=True)))}), impossível dividir estratificadamente.")
        return False, None


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
         logging.error(f"Erro ao dividir os dados: {e}. Verifique se há amostras suficientes de cada classe (pelo menos 2).")
         return False, None

    # Criar estrutura de diretórios
    train_dir = os.path.join(target_base_dir, 'train')
    val_dir = os.path.join(target_base_dir, 'val')

    # Limpa diretórios antigos se existirem
    if os.path.exists(target_base_dir):
        logging.warning(f"Removendo diretório de dataset existente: {target_base_dir}")
        shutil.rmtree(target_base_dir)

    # Cria as pastas das classes dentro de train e val
    img_paths_by_split_class = {'train': {}, 'val': {}}
    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        img_paths_by_split_class['train'][class_name] = []
        img_paths_by_split_class['val'][class_name] = []


    # Função para converter e copiar
    def convert_and_copy(paths, labels, target_set_dir, split_name):
        count = 0
        skipped_conversion = 0
        for img_path, label in zip(paths, labels):
            try:
                # Tenta abrir com Pillow primeiro, fallback para tifffile
                try:
                    img = Image.open(img_path)
                except Exception as pil_error:
                    logging.warning(f"Pillow não conseguiu abrir {img_path} ({pil_error}), tentando tifffile...")
                    img_array = tifffile.imread(img_path)
                    if img_array.ndim == 3:
                        if img_array.shape[2] >= 3:
                           img_array = img_array[:, :, :3] # Assume RGB-like
                        else:
                            img_array = img_array[:,:,0]
                    elif img_array.ndim > 3:
                        logging.warning(f"TIF com muitas dimensões: {img_path}, pegando o primeiro frame.")
                        img_array = img_array[0]
                        if img_array.ndim == 3:
                            if img_array.shape[2] >= 3:
                                img_array = img_array[:, :, :3]
                            else:
                                img_array = img_array[:,:,0]

                    if img_array.dtype != np.uint8:
                        img_min, img_max = np.min(img_array), np.max(img_array)
                        if img_min == img_max: # Evita divisão por zero se a imagem for constante
                            img_array = np.zeros_like(img_array, dtype=np.uint8)
                        else:
                           if np.issubdtype(img_array.dtype, np.floating):
                               img_array = ((img_array - img_min) / (img_max - img_min)) * 255
                           elif img_max > 255: # Ex: 16-bit
                               img_array = (img_array / (img_max / 255.0)) # Mapeia para 0-255
                           # Garantir que não exceda 255 devido a arredondamentos
                           img_array = np.clip(img_array, 0, 255)

                        img_array = img_array.astype(np.uint8)

                    img = Image.fromarray(img_array)

                # Converter para RGB para garantir compatibilidade
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                # Salvar no formato desejado
                base_filename = os.path.basename(img_path)
                new_filename = os.path.splitext(base_filename)[0] + '.' + img_format_out
                target_path = os.path.join(target_set_dir, label, new_filename)
                img.save(target_path)
                # Armazenar o caminho da imagem convertida para análise posterior
                img_paths_by_split_class[split_name][label].append(target_path)
                count += 1
            except Exception as e:
                logging.error(f"Falha ao converter ou salvar {img_path}: {e}")
                traceback.print_exc() # Imprime detalhes do erro de conversão
                skipped_conversion += 1
        logging.info(f"Convertidas e copiadas {count} imagens para {target_set_dir}. Falhas/puladas: {skipped_conversion}")

    # Processar treino e validação
    logging.info("Processando conjunto de Treinamento...")
    convert_and_copy(train_paths, train_labels, train_dir, 'train')
    logging.info("Processando conjunto de Validação...")
    convert_and_copy(val_paths, val_labels, val_dir, 'val')

    logging.info("Preparação do dataset concluída.")
    # Retorna True e o dicionário com os caminhos das imagens convertidas
    return True, img_paths_by_split_class

# --- 2. Funções de Análise de Imagem ---

def calculate_shannon_entropy(img_array_gray):
    """Calcula a entropia de Shannon para uma imagem em tons de cinza."""
    hist = np.histogram(img_array_gray, bins=256, range=(0, 256))[0]
    hist = hist[hist > 0] # Remove bins com contagem zero
    if np.sum(hist) == 0: return 0 # Imagem vazia ou constante
    prob = hist / np.sum(hist)
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

def analyze_image_stats(image_paths_dict, classes):
    """Calcula e imprime luminosidade média e entropia, e plota a entropia."""
    logging.info("Iniciando análise de estatísticas das imagens...")
    stats = {cls: {'luminosity': [], 'entropy': []} for cls in classes}
    all_entropies = {'class': [], 'entropy': []}

    for split in image_paths_dict: # 'train', 'val'
        for cls in classes:
            logging.info(f"Analisando {cls} imagens em {split}...")
            img_counter = 0
            for img_path in image_paths_dict[split][cls]:
                try:
                    img = Image.open(img_path).convert('L') # Abre e converte para Grayscale
                    img_array = np.array(img)

                    # Luminosidade (média dos pixels em grayscale)
                    luminosity = np.mean(img_array)
                    stats[cls]['luminosity'].append(luminosity)

                    # Entropia
                    entropy = calculate_shannon_entropy(img_array)
                    stats[cls]['entropy'].append(entropy)
                    all_entropies['class'].append(cls)
                    all_entropies['entropy'].append(entropy)
                    img_counter += 1

                except Exception as e:
                    logging.warning(f"Falha ao analisar a imagem {img_path}: {e}")
            logging.info(f"Analisadas {img_counter} imagens de {cls} em {split}.")

    print("\n--- Análise de Estatísticas das Imagens ---")
    if not all_entropies['entropy']:
         print("Nenhuma estatística pôde ser calculada (verifique logs de erro).")
         return

    print("\nLuminosidade Média (0=Preto, 255=Branco):")
    for cls in classes:
        if stats[cls]['luminosity']:
            avg_lum = np.mean(stats[cls]['luminosity'])
            print(f"- {cls}: {avg_lum:.2f}")
        else:
            print(f"- {cls}: N/A (nenhuma imagem processada)")

    print("\nEntropia Média (Maior = Mais textura/detalhe):")
    for cls in classes:
         if stats[cls]['entropy']:
            avg_ent = np.mean(stats[cls]['entropy'])
            print(f"- {cls}: {avg_ent:.4f}")
         else:
            print(f"- {cls}: N/A (nenhuma imagem processada)")

    # Plotar Distribuição da Entropia
    if all_entropies['entropy']:
        print("\nGerando gráfico de distribuição de entropia ('entropy_distribution.png')...")
        try:
            plt.figure(figsize=(10, 6))
            sns.kdeplot(data=all_entropies, x='entropy', hue='class', fill=True, common_norm=False)
            plt.title('Distribuição da Entropia da Imagem por Classe')
            plt.xlabel('Entropia de Shannon')
            plt.ylabel('Densidade')
            plt.savefig('entropy_distribution.png')
            plt.close() # Fecha a figura para não exibir interativamente se não for desejado
            print("Gráfico 'entropy_distribution.png' salvo.")
        except Exception as e:
            print(f"Erro ao gerar o gráfico de entropia: {e}")

    print("--- Fim da Análise de Estatísticas ---\n")


# --- 3. Treinamento ---

def train_yolo_classifier(dataset_dir, model_name, epochs, img_size):
    """Treina um modelo de classificação YOLOv8."""
    logging.info(f"Iniciando o treinamento com o modelo: {model_name}")
    logging.info(f"Dataset: {dataset_dir}")
    logging.info(f"Épocas: {epochs}, Tamanho da imagem: {img_size}")
    results_save_dir = None # Inicializa como None

    try:
        # Carrega o modelo de classificação pré-treinado
        model = YOLO(model_name)

        # Treina o modelo
        # A função train retorna um objeto Results ou similar
        results = model.train(
            data=dataset_dir, # Diretório pai contendo 'train' e 'val'
            epochs=epochs,
            imgsz=img_size,
            batch=-1, # -1 para auto-batch (YOLO decide baseado na memória GPU) ou defina um número (ex: 16)
            name='cafe_milho_classifier_run', # Nome do experimento (será salvo em runs/classify/)
            # patience=10 # Opcional: parar cedo se não houver melhora por X épocas
        )
        logging.info("Treinamento YOLO concluído com sucesso (fase de execução).")
        # O caminho é geralmente acessado via atributo do objeto retornado
        # Tentar obter o diretório de salvamento. A API pode variar ligeiramente entre versões.
        # Em versões recentes, o diretório pode estar em results.save_dir
        if hasattr(results, 'save_dir'):
             results_save_dir = results.save_dir
             logging.info(f"Resultados do treinamento salvos em: {results_save_dir}")
        else:
             logging.warning("Não foi possível determinar automaticamente o diretório de salvamento dos resultados do treino.")
             # Tentar um caminho padrão (pode não ser exato)
             results_save_dir = os.path.join('runs', 'classify', 'cafe_milho_classifier_run')
             logging.warning(f"Assumindo que os resultados estão em: {results_save_dir} (verifique manualmente)")


    except Exception as e:
        # Este bloco captura qualquer erro que ocorra DENTRO do try, incluindo erros no model.train()
        logging.error(f"ERRO CRÍTICO DURANTE O TREINAMENTO YOLO: {e}")
        logging.error("Verifique a mensagem de erro acima e o traceback completo.")
        traceback.print_exc() # Imprime o traceback completo do erro original
        results_save_dir = None # Garante que retornará None em caso de erro

    return results_save_dir # Retorna o caminho ou None

# --- 4. Execução Principal ---

if __name__ == "__main__":
    # Garante que o diretório de origem existe
    if not os.path.isdir(SOURCE_IMAGE_DIR):
         logging.error(f"Diretório de origem das imagens não encontrado: {SOURCE_IMAGE_DIR}")
         logging.error("Por favor, edite a variável SOURCE_IMAGE_DIR no script.")
    else:
        # 1. Preparar os dados
        dataset_ready, converted_image_paths = prepare_datasets(
            SOURCE_IMAGE_DIR, DATASET_BASE_DIR, TRAIN_RATIO, CLASSES, IMAGE_FORMAT_OUT
        )

        if dataset_ready and converted_image_paths:
            # 2. Analisar estatísticas das imagens (luminosidade, entropia)
            analyze_image_stats(converted_image_paths, CLASSES)

            # 3. Treinar o modelo
            logging.info("Prosseguindo para o treinamento do modelo YOLO...")
            results_path = train_yolo_classifier(DATASET_BASE_DIR, MODEL_TO_USE, NUM_EPOCHS, IMAGE_SIZE)

            # 4. Analisar Resultados do Treino (se o treino ocorreu bem)
            if results_path: # Verifica se o treinamento retornou um caminho válido (não None)
                logging.info(f"\n--- Análise Pós-Treino ---")
                logging.info(f"O modelo treinado e os resultados estão em: {results_path}")
                logging.info("Verifique os arquivos gerados lá, como:")
                logging.info(f"- results.png / results.csv: Gráficos e dados de desempenho.")
                logging.info(f"- confusion_matrix.png: Matriz de confusão (importante para ver erros entre Cafe/Milho).")
                logging.info(f"- weights/best.pt: O melhor modelo treinado.")
                logging.info(f"- Arquivo de plot 'entropy_distribution.png' foi salvo no diretório atual do script.")

                logging.info("\nPara tentar entender as 'diferenças' que o modelo aprendeu:")
                logging.info("1. Compare as estatísticas de LUMINOSIDADE e ENTROPIA médias calculadas.")
                logging.info("   - Uma diferença significativa pode ser um indicador que o modelo também pode usar.")
                logging.info("2. Analise o gráfico 'entropy_distribution.png'.")
                logging.info("   - Se as distribuições de entropia para Café e Milho forem bem separadas, a textura é um bom diferenciador.")
                logging.info("   - Se houver muita sobreposição, a entropia sozinha pode não ser suficiente.")
                logging.info("3. Analise a 'confusion_matrix.png' no diretório de resultados: Veja se o modelo confunde mais 'Cafe' com 'Milho' ou vice-versa.")
                logging.info("4. Olhe as imagens no conjunto de validação ({}/val/...) que foram classificadas INCORRETAMENTE.")
                logging.info("   - O que há de visualmente diferente ou semelhante nessas imagens que pode ter confundido o modelo?")
                logging.info("   - Há grãos de milho muito sutis? A iluminação é diferente? A textura é ambígua?")
                logging.info("5. (Avançado) Pesquise por 'Grad-CAM YOLOv8 classification' para encontrar tutoriais sobre como gerar mapas de calor.")

            else:
                # Esta mensagem aparece se train_yolo_classifier retornou None
                logging.error("O treinamento do modelo YOLO falhou ou foi interrompido.")
                logging.error("Verifique os logs de erro acima para detalhes sobre o problema ocorrido durante o treinamento.")
        else:
            logging.error("Falha na preparação do dataset. O treinamento não pode continuar.")
            logging.error("Verifique os logs para erros na leitura, conversão ou divisão dos arquivos.")