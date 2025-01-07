from ultralytics import YOLO

# Carrega o modelo e o conjunto de dados de validação
model = YOLO("weights/best.pt")
results = model.val(data="path/to/dataset.yaml")  

# Imprime a avaliação
print(f"mAP50: {results['metrics/mAP_0.5']}")
print(f"mAP50-95: {results['metrics/mAP_0.5:0.95']}")