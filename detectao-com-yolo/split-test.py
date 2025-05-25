import os
import shutil
import random

def split_dataset(original_class_dir, train_dir, val_dir, split_ratio=0.8):
    images = os.listdir(original_class_dir)
    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)

    train_images = images[:split_idx]
    val_images = images[split_idx:]

    for img in train_images:
        src = os.path.join(original_class_dir, img)
        dst = os.path.join(train_dir, img)
        shutil.copyfile(src, dst)  # Copia para treino

    for img in val_images:
        src = os.path.join(original_class_dir, img)
        dst = os.path.join(val_dir, img)
        shutil.copyfile(src, dst)  # Copia para validação

# Classes do dataset
classes = ['milho', 'cafe']

for cls in classes:
    os.makedirs(f'./dataset/train/{cls}', exist_ok=True)
    os.makedirs(f'./dataset/val/{cls}', exist_ok=True)

# Aplicar split para cada classe
for cls in classes:
    split_dataset(
        original_class_dir=rf'/home/pedro-henrique/Documentos/detector-cafe/detector-impurezas-cafe/dataset/all/{cls}',
        train_dir=f'./dataset/train/{cls}',
        val_dir=f'./dataset/val/{cls}',
        split_ratio=0.8  # 80% treino, 20% validação
    )

