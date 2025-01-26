import cv2
from ultralytics import YOLO

# Caminho para o modelo treinado
model_path = 'pupil_tracker.pt'
model = YOLO(model_path)

# Número máximo de vídeos (substitua por x ou ajuste conforme necessário)
num_videos = 5  # Por exemplo, se você tem pupil_0 a pupil_9

for i in range(num_videos):
    # Caminho para o vídeo de entrada
    video_path = f'pupil_{i}.mp4'  # Substitua pela nomenclatura dos seus vídeos
    output_path = f'output_pupil_track_{i}.mp4'  # Caminho para salvar o vídeo com os resultados

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

        # Realizar o rastreamento no frame atual, persistindo entre os frames
        results = model.track(frame, persist=True, conf=0.5, iou=0.5)  # Ajuste 'conf' e 'iou' conforme necessário

        # Visualizar e processar as detecções no frame
        annotated_frame = frame.copy()  # Cópia do frame para anotações
        for result in results:
            if hasattr(result, "boxes"):
                for box in result.boxes.xyxy:
                    bbox = box.cpu().numpy().astype(int)
                    x_min, y_min, x_max, y_max = bbox

                    # Desenhar a caixa delimitadora (bbox)
                    cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    # Adicionar um identificador de rastreamento, se disponível
                    if hasattr(result, "id") and result.id is not None:
                        tracker_id = result.id
                        cv2.putText(
                            annotated_frame, f'ID: {tracker_id}',
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                        )

        # Exibir o frame com as detecções
        cv2.imshow("YOLO Tracking com Rastreamento", annotated_frame)

        # Escrever o frame no arquivo de saída
        out.write(annotated_frame)

        # Sair com a tecla 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Liberar recursos para o vídeo atual
    cap.release()
    out.release()
    cv2.destroyAllWindows()

print("Processamento concluído para todos os vídeos.")
