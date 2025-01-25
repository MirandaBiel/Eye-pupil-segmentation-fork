import cv2
from ultralytics import YOLO
import torch
import numpy as np

# Caminho para o modelo treinado
model_path = 'pupil_tracker.pt'
model = YOLO(model_path)

# Número máximo de vídeos (substitua por x ou ajuste conforme necessário)
num_videos = 5  # Por exemplo, se você tem pupil_0 a pupil_9

for i in range(num_videos):
    # Caminho para o vídeo de entrada
    video_path = f'pupil_{i}.mp4'  # Substitua pela nomenclatura dos seus vídeos
    output_path = f'output_pupil_{i}.mp4'  # Caminho para salvar o vídeo com os resultados

    # Carregar o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Não foi possível abrir o vídeo: {video_path}")
        continue

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
                for mask, bbox in zip(result.masks.data, result.boxes.xyxy):
                    # Extrair a máscara e o bbox
                    mask_np = mask.cpu().numpy()
                    bbox = bbox.cpu().numpy().astype(int)

                    # Coordenadas da caixa delimitadora (bbox)
                    x_min, y_min, x_max, y_max = bbox

                    # Redimensionar a máscara para o tamanho do frame
                    mask_resized = cv2.resize(mask_np, (frame.shape[1], frame.shape[0]))

                    # Aplicar a máscara somente dentro da bbox
                    mask_cropped = mask_resized[y_min:y_max, x_min:x_max]

                    # Criar uma máscara binária colorida
                    mask_colored = cv2.applyColorMap((mask_cropped * 255).astype('uint8'), cv2.COLORMAP_JET)

                    # Combinar a máscara com o frame na área do bbox
                    frame[y_min:y_max, x_min:x_max] = cv2.addWeighted(
                        frame[y_min:y_max, x_min:x_max], 0.7, mask_colored, 0.3, 0
                    )

                    # Desenhar a caixa delimitadora (bbox)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        # Mostrar o frame com as detecções
        cv2.imshow("Detecção com Máscara e BBox", frame)

        # Escrever o frame no arquivo de saída
        out.write(frame)

        # Sair com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos para o vídeo atual
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print("Processamento concluído para todos os vídeos.")
