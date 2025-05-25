from ultralytics import YOLO

# Carregar modelo de classificação
model = YOLO('yolo11n-cls.pt')

# Treinar
model.train(
    data='/home/pedro-henrique/Documentos/detector-cafe/detector-impurezas-cafe/detectao-com-yolo/dataset',
    epochs=50,
    imgsz=224
)

