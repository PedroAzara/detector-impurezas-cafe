import os
import shutil

# Caminho da pasta com todas as imagens
src_dir = r'/home/pedro-henrique/Documentos/detector-cafe/detector-impurezas-cafe/imagens/imagens-02-04-25-png'

# Pastas destino
dest_dirs = {
    'milho': './dataset/milho',
    'cafe': './dataset/cafe'
}

# Criar pastas destino se não existirem
for d in dest_dirs.values():
    os.makedirs(d, exist_ok=True)

# Separar imagens
for img in os.listdir(src_dir):
    if 'milho' in img.lower():
        shutil.move(os.path.join(src_dir, img), os.path.join(dest_dirs['milho'], img))
    elif 'cafe' in img.lower():
        shutil.move(os.path.join(src_dir, img), os.path.join(dest_dirs['cafe'], img))
