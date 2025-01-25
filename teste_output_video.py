import cv2
from ultralytics import YOLO
import torch

# Caminho para o modelo treinado
model_path = 'pupil_tracker.pt'
model = YOLO(model_path)

# Caminho para o vídeo de entrada
video_path = 'pupil_2.mp4'  # Substitua pelo caminho do vídeo
output_path = 'pupil_2_output.mp4'  # Caminho para salvar o vídeo com as máscaras

# Carregar o vídeo
cap = cv2.VideoCapture(video_path)

# Obter informações do vídeo
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Inicializar o writer para salvar o vídeo
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Processar o vídeo frame a frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Realizar a predição no frame atual
    results = model.predict(source=frame, save=False, show=False)

    # Iterar sobre as detecções no frame
    for result in results:
        if hasattr(result, "masks") and result.masks:  # Verificar se há máscaras
            for mask in result.masks.data:
                # Converter a máscara para numpy
                mask_np = mask.cpu().numpy()

                # Redimensionar a máscara para o tamanho do frame
                mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))

                # Converter a máscara para 8 bits
                mask_resized = (mask_resized * 255).astype('uint8')

                # Criar uma máscara colorida
                mask_colored = cv2.applyColorMap(mask_resized, cv2.COLORMAP_JET)

                # Combinar o frame original com a máscara colorida
                frame = cv2.addWeighted(frame, 0.7, mask_colored, 0.3, 0)

    # Mostrar o frame com a máscara
    cv2.imshow("Detecção com Máscara", frame)

    # Escrever o frame no arquivo de saída
    out.write(frame)

    # Sair com a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
out.release()
cv2.destroyAllWindows()
