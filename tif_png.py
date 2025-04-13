# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 12:53:52 2025

@author: ramoe
"""

import os
import numpy as np # Precisaremos do NumPy para escalonamento se necessário
from PIL import Image, ImageOps # ImageOps pode ser útil
from tqdm import tqdm

# Desativa o limite de descompressão do Pillow (mantém)
Image.MAX_IMAGE_PIXELS = None

def convert_tif_to_png(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    print(f"Pasta de saída: {output_folder}")

    try:
        files = os.listdir(input_folder)
    except FileNotFoundError:
        print(f"Erro Crítico: Pasta de entrada não encontrada: {input_folder}")
        return
    except Exception as e:
        print(f"Erro ao listar arquivos em {input_folder}: {e}")
        return

    print(f"Procurando por arquivos .tif/.tiff em '{input_folder}'...")
    tif_files = [f for f in files if f.lower().endswith((".tif", ".tiff"))]

    if not tif_files:
        print("Nenhum arquivo .tif ou .tiff encontrado.")
        return

    print(f"Encontrados {len(tif_files)} arquivos TIFF.")
    converted_count = 0
    error_count = 0
    mode_summary = {} # Dicionário para contar os modos encontrados

    for filename in tqdm(tif_files, desc="Convertendo TIF para PNG"):
        input_path = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        output_filename = f"{base_name}.png"
        output_path = os.path.join(output_folder, output_filename)

        try:
            with Image.open(input_path) as img:
                # --- Diagnóstico Crucial ---
                original_mode = img.mode
                print(f"\nProcessando: {filename}, Modo Original: {original_mode}, Tamanho: {img.size}")

                # Atualiza contagem de modos
                mode_summary[original_mode] = mode_summary.get(original_mode, 0) + 1

                img_to_save = None # Começa sem imagem para salvar

                # --- Tentativas de Conversão ---

                # 1. Tentar salvar diretamente (se for modo compatível como L, RGB, RGBA)
                if original_mode in ('L', 'RGB', 'RGBA', 'P'):
                     try:
                         # Para paletas, converter para RGB/RGBA é mais seguro
                         if original_mode == 'P':
                             print(f"  Convertendo Paleta (P) para {'RGBA' if 'transparency' in img.info else 'RGB'}")
                             img_to_save = img.convert('RGBA' if 'transparency' in img.info else 'RGB')
                         else:
                             img_to_save = img # Usar a imagem como está
                         img_to_save.save(output_path, "PNG")
                         print(f"  Salvo diretamente/convertido de P.")
                         converted_count += 1
                         continue # Vai para o próximo arquivo
                     except Exception as e_direct:
                         print(f"  Falha ao salvar diretamente/converter de P: {e_direct}. Tentando outras conversões...")
                         img_to_save = None # Reseta

                # 2. Lidar com modos de alta profundidade de bits (Integer)
                #    Modos comuns: 'I;16', 'I;16B', 'I;16L', 'I', 'I;32', etc.
                if original_mode.startswith('I'):
                    print(f"  Modo Integer detectado ({original_mode}). Tentando escalonar para 8-bit (L).")
                    # Converte para NumPy array para obter min/max e escalar
                    np_img = np.array(img)
                    min_val, max_val = np_img.min(), np_img.max()
                    print(f"    Valores Min/Max: {min_val}/{max_val}")

                    if max_val > min_val:
                        # Escalonar para 0-255
                        scaled_img = ((np_img - min_val) * (255.0 / (max_val - min_val))).astype(np.uint8)
                        img_to_save = Image.fromarray(scaled_img, mode='L') # Salva como Grayscale 8-bit
                        print("    Escalonado para 8-bit 'L'.")
                    else: # Imagem constante
                         img_to_save = Image.fromarray(np.full(np_img.shape, int(min_val > 0) * 255, dtype=np.uint8) , mode='L')
                         print("    Imagem com valor constante, convertida para L (branco ou preto).")

                # 3. Lidar com modo Float ('F')
                elif original_mode == 'F':
                    print("  Modo Float ('F') detectado. Tentando escalonar para 8-bit (L).")
                    np_img = np.array(img)
                    min_val, max_val = np_img.min(), np_img.max()
                    print(f"    Valores Min/Max Float: {min_val}/{max_val}")
                    if max_val > min_val:
                         # Assumindo que os dados são para serem mapeados para 0-255
                         scaled_img = np.clip(((np_img - min_val) / (max_val - min_val + 1e-8)) * 255, 0, 255).astype(np.uint8)
                         img_to_save = Image.fromarray(scaled_img, mode='L')
                         print("    Escalonado para 8-bit 'L'.")
                    else: # Imagem constante
                         img_to_save = Image.fromarray(np.full(np_img.shape, int(min_val > 0) * 255, dtype=np.uint8) , mode='L')
                         print("    Imagem Float com valor constante, convertida para L.")


                # 4. Lidar com CMYK
                elif original_mode == 'CMYK':
                    print("  Modo CMYK detectado. Convertendo para RGB.")
                    img_to_save = img.convert('RGB')

                # 5. Outros modos (tentativa genérica de conversão para RGB)
                #    Pode não funcionar bem para todos os modos!
                elif img_to_save is None: # Se nenhuma das condições anteriores foi satisfeita
                    print(f"  Modo '{original_mode}' não tratado especificamente. Tentando converter para RGB.")
                    try:
                        img_to_save = img.convert('RGB')
                    except Exception as e_conv:
                         print(f"  Falha ao converter modo '{original_mode}' para RGB: {e_conv}")
                         error_count += 1
                         continue # Pula para o próximo arquivo


                # --- Salvar a imagem processada (se alguma conversão funcionou) ---
                if img_to_save:
                    try:
                        img_to_save.save(output_path, "PNG")
                        converted_count += 1
                    except Exception as e_save:
                         print(f"  ERRO FINAL ao salvar {filename} após conversão: {e_save}")
                         error_count += 1
                elif original_mode not in ('L', 'RGB', 'RGBA', 'P'): # Se não entrou em nenhum if/elif de conversão bem-sucedida
                    print(f"  Não foi possível determinar como converter o modo '{original_mode}' para PNG.")
                    error_count += 1


        except FileNotFoundError:
            print(f"\nErro: Arquivo não encontrado: {input_path}")
            error_count += 1
        except Exception as e:
            print(f"\nErro GERAL ao processar {filename}: {e}")
            error_count += 1

    print("\n--- Conversão Concluída ---")
    print("Resumo dos Modos de Imagem TIFF encontrados:")
    for mode, count in mode_summary.items():
        print(f" - Modo '{mode}': {count} arquivos")
    print(f"Arquivos TIFF processados: {len(tif_files)}")
    print(f"Arquivos convertidos com sucesso para PNG: {converted_count}")
    print(f"Arquivos que falharam ou não puderam ser convertidos: {error_count}")
    print(f"Arquivos PNG foram salvos em: {output_folder}")

# --- CONFIGURAÇÃO ---
# !!! IMPORTANTE: Modifique estes dois caminhos !!!




pasta_contendo_os_tiffs = r"C:\Users\pedro\OneDrive\Documentos\projetos\imagens\imagens-cafe-tiff\02-04-25"
# Exemplo: pasta_contendo_os_tiffs = r"C:\Users\SeuUsuario\Imagens\TIFs"
pasta_onde_salvar_os_pngs = r"C:\Users\pedro\OneDrive\Documentos\projetos\imagens\imagens-cafe-png\02-04-25"
# Exemplo: pasta_onde_salvar_os_pngs = r"C:\Users\SeuUsuario\Imagens\PNGs"

# --- EXECUÇÃO ---
if __name__ == "__main__":
    if "Caminho\\Para\\Sua" in pasta_contendo_os_tiffs or \
       "Caminho\\Para\\Sua" in pasta_onde_salvar_os_pngs:
        print("*" * 60)
        print("!! ATENÇÃO !! Edite 'pasta_contendo_os_tiffs' e 'pasta_onde_salvar_os_pngs'")
        print("*" * 60)
    else:
        # Importar NumPy aqui se a lógica de escalonamento for usada
        try:
            import numpy as np
            print("NumPy importado com sucesso.")
        except ImportError:
            print("*"*60)
            print("ERRO: NumPy não está instalado. Necessário para escalonamento.")
            print("Instale com: pip install numpy")
            print("*"*60)
            exit() # Interrompe se NumPy não estiver disponível

        convert_tif_to_png(pasta_contendo_os_tiffs, pasta_onde_salvar_os_pngs)


#