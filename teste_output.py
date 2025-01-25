from ultralytics import YOLO

# Caminho para o modelo treinado
model_path = 'pupil_tracker.pt'  # Certifique-se de que o arquivo está no local correto

# Carregar o modelo
model = YOLO(model_path)

# Caminho para a imagem de teste
image_path = 'eye1.png'

# Realizar a predição
results = model.predict(source=image_path)

# Iterar sobre os resultados
for result in results:
    print("Classes detectadas:", result.names)  # Nome das classes detectadas

    # Acessar as caixas delimitadoras
    for bbox in result.boxes.xyxy:  # Coordenadas no formato [x_min, y_min, x_max, y_max]
        print(f"Caixa delimitadora (bbox): {bbox.tolist()}")

    # Acessar as pontuações de confiança
    for conf in result.boxes.conf:
        print(f"Confiança: {conf.item()}")

    # Acessar os rótulos (índice das classes detectadas)
    for cls in result.boxes.cls:
        print(f"Classe detectada (índice): {int(cls.item())}")

    # Acessar as máscaras (apenas para segmentação)
    if hasattr(result, "masks") and result.masks:
        for mask in result.masks.data:  # Máscaras binárias para segmentação
            print(f"Máscara (formato {mask.shape}): {mask}")
