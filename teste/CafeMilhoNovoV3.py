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
import pandas as pd # Para dataframes
import cv2 # Para conversão HSV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- Configurações ---
SOURCE_IMAGE_DIR = 'tif' # !!! MUDE ISSO !!!
DATASET_BASE_DIR = './cafe_milho_dataset_classif' # Onde o dataset formatado será criado
CLUSTER_PLOTS_DIR = './cluster_plots' # Pasta para salvar os gráficos de cluster
TRAIN_RATIO = 0.7
IMAGE_FORMAT_OUT = 'png' # Formato para conversão (png ou jpg)
MODEL_TO_USE = 'yolov8m-cls.pt' # Modelo inicial (pequeno e rápido)
NUM_EPOCHS = 50 # Número de épocas de treinamento (ajuste conforme necessário)
IMAGE_SIZE = 128 # Tamanho da imagem para treino (ajuste conforme necessário)

CLASSES = ['Cafe', 'Milho']

# Configurar logging para ver o que está acontecendo
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 1. Funções de Preparação de Dados (mantidas como antes) ---
def get_label_from_filename(filename):
    """Extrai o rótulo ('Cafe' ou 'Milho') do nome do arquivo."""
    basename = os.path.basename(filename).lower()
    if basename.startswith('cafeara'): return 'Cafe'
    elif basename.startswith('milho'): return 'Milho'
    else:
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
        logging.error("Nenhum arquivo TIF encontrado no diretório de origem.")
        return False, None

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
        logging.error(f"Imagens insuficientes ou classes ausentes. Rótulos encontrados: {set(labels)}")
        return False, None

    min_samples_per_class = min(labels.count(c) for c in set(labels)) if labels else 0
    required_min_samples = 2 # Necessário para train_test_split com stratify
    if min_samples_per_class < required_min_samples:
         logging.error(f"Pelo menos uma classe tem menos de {required_min_samples} amostras ({dict(zip(*np.unique(labels, return_counts=True)))}), impossível dividir.")
         return False, None

    logging.info(f"Total de imagens com rótulos válidos: {len(image_paths)}")
    logging.info(f"Distribuição inicial - Cafe: {labels.count('Cafe')}, Milho: {labels.count('Milho')}")

    try:
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            image_paths, labels, train_size=train_ratio, random_state=42, stratify=labels
        )
        logging.info(f"Divisão - Treino: {len(train_paths)} img, Validação: {len(val_paths)} img")
    except ValueError as e:
         logging.error(f"Erro ao dividir os dados: {e}. Verifique a distribuição das classes.")
         return False, None

    train_dir = os.path.join(target_base_dir, 'train')
    val_dir = os.path.join(target_base_dir, 'val')
    if os.path.exists(target_base_dir):
        logging.warning(f"Removendo diretório de dataset existente: {target_base_dir}")
        shutil.rmtree(target_base_dir)

    img_paths_by_split_class = {'train': {}, 'val': {}}
    for class_name in classes:
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        img_paths_by_split_class['train'][class_name] = []
        img_paths_by_split_class['val'][class_name] = []

    def convert_and_copy(paths, labels, target_set_dir, split_name):
        count = 0
        skipped = 0
        for img_path, label in zip(paths, labels):
            try:
                try: img = Image.open(img_path)
                except Exception:
                    logging.warning(f"Pillow falhou com {img_path}, tentando tifffile...")
                    img_array = tifffile.imread(img_path)
                    if img_array.ndim == 3: img_array = img_array[:,:,:3] if img_array.shape[2]>=3 else img_array[:,:,0]
                    elif img_array.ndim > 3: img_array = img_array[0]; img_array = img_array[:,:,:3] if img_array.ndim==3 and img_array.shape[2]>=3 else img_array[:,:,0] if img_array.ndim==3 else img_array
                    if img_array.dtype != np.uint8:
                        img_min, img_max = np.min(img_array), np.max(img_array)
                        if img_min == img_max: img_array = np.zeros_like(img_array, dtype=np.uint8)
                        else:
                           if np.issubdtype(img_array.dtype, np.floating): img_array = ((img_array - img_min) / (img_max - img_min)) * 255
                           elif img_max > 255: img_array = (img_array / (img_max / 255.0))
                           img_array = np.clip(img_array, 0, 255).astype(np.uint8)
                    img = Image.fromarray(img_array)

                if img.mode != 'RGB': img = img.convert('RGB')
                base_filename = os.path.basename(img_path)
                new_filename = os.path.splitext(base_filename)[0] + '.' + img_format_out
                target_path = os.path.join(target_set_dir, label, new_filename)
                img.save(target_path)
                img_paths_by_split_class[split_name][label].append(target_path)
                count += 1
            except Exception as e:
                logging.error(f"Falha ao processar {img_path}: {e}")
                # traceback.print_exc() # Descomente para debug detalhado da conversão
                skipped += 1
        logging.info(f"Convertidas e copiadas {count} imagens para {target_set_dir}. Falhas/puladas: {skipped}")

    logging.info("Processando conjunto de Treinamento...")
    convert_and_copy(train_paths, train_labels, train_dir, 'train')
    logging.info("Processando conjunto de Validação...")
    convert_and_copy(val_paths, val_labels, val_dir, 'val')

    logging.info("Preparação do dataset concluída.")
    return True, img_paths_by_split_class

# --- 2. Funções de Análise de Imagem (Modificada) ---

def calculate_shannon_entropy(img_array_gray):
    """Calcula a entropia de Shannon para uma imagem em tons de cinza."""
    hist = np.histogram(img_array_gray, bins=256, range=(0, 256))[0]
    hist = hist[hist > 0]
    if np.sum(hist) == 0: return 0
    prob = hist / np.sum(hist)
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

def analyze_image_features(image_paths_dict, classes):
    """Calcula características (luminosidade, entropia, contraste, saturação) para cada imagem."""
    logging.info("Iniciando cálculo de características das imagens...")
    feature_data = []

    for split in image_paths_dict: # 'train', 'val'
        for cls in classes:
            logging.info(f"Calculando características de {cls} imagens em {split}...")
            img_counter = 0
            for img_path in image_paths_dict[split][cls]:
                try:
                    # Carrega a imagem em RGB (para saturação) e Grayscale (para outras)
                    img_pil_rgb = Image.open(img_path).convert('RGB')
                    img_pil_gray = img_pil_rgb.convert('L')
                    img_array_gray = np.array(img_pil_gray)

                    # Usa OpenCV para conversão HSV (mais robusto)
                    img_cv2_rgb = np.array(img_pil_rgb)
                    img_cv2_hsv = cv2.cvtColor(img_cv2_rgb, cv2.COLOR_RGB2HSV)

                    # Luminosidade (Média de pixels Grayscale)
                    luminosity = np.mean(img_array_gray)

                    # Entropia (Baseado em Grayscale)
                    entropy = calculate_shannon_entropy(img_array_gray)

                    # Contraste (Desvio Padrão de pixels Grayscale)
                    contrast = np.std(img_array_gray)

                    # Saturação Média (Canal S do HSV)
                    # O canal S em OpenCV vai de 0 a 255
                    saturation = np.mean(img_cv2_hsv[:, :, 1])

                    feature_data.append({
                        'path': img_path,
                        'class': cls,
                        'split': split,
                        'Luminosity': luminosity,
                        'Entropy': entropy,
                        'Contrast': contrast,
                        'Saturation': saturation
                    })
                    img_counter += 1

                except Exception as e:
                    logging.warning(f"Falha ao calcular características da imagem {img_path}: {e}")
            logging.info(f"Características calculadas para {img_counter} imagens de {cls} em {split}.")

    if not feature_data:
        logging.error("Nenhuma característica de imagem pôde ser calculada.")
        return None

    df_features = pd.DataFrame(feature_data)
    logging.info("Cálculo de características concluído.")
    return df_features

def plot_feature_clusters(df_features, output_dir):
    """Gera e salva gráficos de dispersão e PCA para visualizar clusters."""
    if df_features is None or df_features.empty:
        logging.warning("DataFrame de características vazio, pulando plotagem de clusters.")
        return

    logging.info(f"Gerando gráficos de cluster e salvando em: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    features_to_plot = ['Luminosity', 'Entropy', 'Contrast', 'Saturation']

    # --- Gráficos de Dispersão (Scatter Plots) para pares de características ---
    pairs_to_plot = [
        ('Luminosity', 'Entropy'),
        ('Luminosity', 'Contrast'),
        ('Entropy', 'Contrast'),
        ('Luminosity', 'Saturation')
    ]

    for x_feat, y_feat in pairs_to_plot:
        if x_feat in df_features.columns and y_feat in df_features.columns:
            plt.figure(figsize=(10, 7))
            sns.scatterplot(data=df_features, x=x_feat, y=y_feat, hue='class', style='split', alpha=0.7)
            plt.title(f'Cluster por {x_feat} vs {y_feat}')
            plt.tight_layout()
            plot_filename = os.path.join(output_dir, f'scatter_{x_feat}_vs_{y_feat}.png')
            try:
                plt.savefig(plot_filename)
                logging.info(f"Salvo: {plot_filename}")
            except Exception as e:
                logging.error(f"Erro ao salvar {plot_filename}: {e}")
            plt.close()
        else:
            logging.warning(f"Pulando plot {x_feat} vs {y_feat}, uma das colunas não encontrada.")


    # --- Análise de Componentes Principais (PCA) ---
    logging.info("Executando PCA para visualização 2D...")
    pca_features = df_features[features_to_plot].dropna() # Usa apenas linhas sem NaN nas features
    labels_for_pca = df_features.loc[pca_features.index, 'class'] # Pega os labels correspondentes

    if pca_features.empty:
        logging.warning("Não há dados suficientes para PCA após remover NaNs.")
        return

    # Escalar os dados antes do PCA
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(pca_features)

    # Aplicar PCA para reduzir a 2 componentes
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(scaled_features)

    # Criar DataFrame com os resultados do PCA
    df_pca = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
    df_pca['class'] = labels_for_pca.values # Adiciona a classe de volta

    # Plotar os resultados do PCA
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=df_pca, x='PC1', y='PC2', hue='class', alpha=0.8)
    plt.title('Cluster das Imagens por PCA (2 Componentes Principais)')
    plt.xlabel(f'Componente Principal 1 ({pca.explained_variance_ratio_[0]*100:.1f}% variância)')
    plt.ylabel(f'Componente Principal 2 ({pca.explained_variance_ratio_[1]*100:.1f}% variância)')
    plt.tight_layout()
    pca_plot_filename = os.path.join(output_dir, 'pca_cluster_2D.png')
    try:
        plt.savefig(pca_plot_filename)
        logging.info(f"Salvo: {pca_plot_filename}")
    except Exception as e:
         logging.error(f"Erro ao salvar {pca_plot_filename}: {e}")
    plt.close()

    logging.info("Geração de gráficos de cluster concluída.")


# --- 3. Treinamento (Mantido como antes) ---
def train_yolo_classifier(dataset_dir, model_name, epochs, img_size):
    """Treina um modelo de classificação YOLOv8."""
    logging.info(f"Iniciando o treinamento com o modelo: {model_name}")
    results_save_dir = None
    try:
        model = YOLO(model_name)
        results = model.train(
            data=dataset_dir, epochs=epochs, imgsz=img_size, batch=-1,
            name='cafe_milho_classifier_run'#, patience=10 # Opcional
        )
        logging.info("Treinamento YOLO concluído com sucesso.")
        if hasattr(results, 'save_dir'):
             results_save_dir = results.save_dir
             logging.info(f"Resultados do treinamento salvos em: {results_save_dir}")
        else:
             logging.warning("Não foi possível determinar o diretório de save dos resultados.")
             results_save_dir = os.path.join('runs', 'classify', 'cafe_milho_classifier_run')
             logging.warning(f"Assumindo que os resultados estão em: {results_save_dir}")
    except Exception as e:
        logging.error(f"ERRO CRÍTICO DURANTE O TREINAMENTO YOLO: {e}")
        traceback.print_exc()
        results_save_dir = None
    return results_save_dir

# --- 4. Execução Principal (Modificada) ---
if __name__ == "__main__":
    if not os.path.isdir(SOURCE_IMAGE_DIR):
         logging.error(f"Diretório de origem não encontrado: {SOURCE_IMAGE_DIR}")
    else:
        # 1. Preparar os dados
        dataset_ready, converted_image_paths = prepare_datasets(
            SOURCE_IMAGE_DIR, DATASET_BASE_DIR, TRAIN_RATIO, CLASSES, IMAGE_FORMAT_OUT
        )

        if dataset_ready and converted_image_paths:
            # 2. Calcular características das imagens
            df_image_features = analyze_image_features(converted_image_paths, CLASSES)

            # 3. Plotar clusters das características (se calculadas)
            if df_image_features is not None and not df_image_features.empty:
                print("\n--- Resumo das Características Calculadas ---")
                print(df_image_features.groupby('class')[['Luminosity', 'Entropy', 'Contrast', 'Saturation']].mean())
                print("--------------------------------------------")
                # Salva o dataframe com as features calculadas
                features_csv_path = os.path.join(CLUSTER_PLOTS_DIR, 'image_features.csv')
                try:
                    os.makedirs(CLUSTER_PLOTS_DIR, exist_ok=True)
                    df_image_features.to_csv(features_csv_path, index=False)
                    logging.info(f"Características das imagens salvas em: {features_csv_path}")
                except Exception as e:
                    logging.error(f"Não foi possível salvar o CSV de características: {e}")

                plot_feature_clusters(df_image_features, CLUSTER_PLOTS_DIR)
            else:
                 logging.warning("Não foi possível gerar o DataFrame de características, pulando análise e plots de cluster.")


            # 4. Treinar o modelo YOLO
            logging.info("Prosseguindo para o treinamento do modelo YOLO...")
            results_path = train_yolo_classifier(DATASET_BASE_DIR, MODEL_TO_USE, NUM_EPOCHS, IMAGE_SIZE)

            # 5. Analisar Resultados do Treino (se ocorreu bem)
            if results_path:
                logging.info(f"\n--- Análise Pós-Treino ---")
                logging.info(f"O modelo treinado e os resultados estão em: {results_path}")
                logging.info("Verifique os arquivos gerados lá (confusion_matrix.png, results.csv, etc.).")
                logging.info(f"Gráficos de cluster das características foram salvos em: {CLUSTER_PLOTS_DIR}")

                logging.info("\nPara entender as 'diferenças' encontradas:")
                logging.info("1. Analise os GRÁFICOS DE CLUSTER salvos na pasta 'cluster_plots':")
                logging.info("   - Os scatter plots mostram como as classes se separam usando pares de características (Luminosidade vs Entropia, etc.).")
                logging.info("   - O gráfico PCA mostra a separação geral usando uma combinação linear das características.")
                logging.info("   - Se os pontos de 'Cafe' e 'Milho' estiverem bem separados em algum gráfico, essas características são bons diferenciadores.")
                logging.info("2. Compare as MÉDIAS das características impressas no console e salvas em 'image_features.csv'.")
                logging.info("3. Analise a MATRIZ DE CONFUSÃO ('confusion_matrix.png') do treino YOLO para ver os erros do modelo.")
                logging.info("4. Compare visualmente as imagens classificadas incorretamente com as corretamente classificadas.")
                logging.info("5. Considere se as características visuais (capturadas nos gráficos) coincidem com o que o modelo YOLO parece estar usando (baseado nos erros e acertos).")

            else:
                logging.error("O treinamento do modelo YOLO falhou ou foi interrompido.")
                logging.error("Verifique os logs de erro acima para detalhes.")
        else:
            logging.error("Falha na preparação do dataset. Análise de características e treinamento não podem continuar.")