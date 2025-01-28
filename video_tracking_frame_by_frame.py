import os
import cv2
from ultralytics import YOLO

# Caminho para o modelo treinado
model_path = "pupil_tracker.pt"
model = YOLO(model_path)

# Pasta contendo os vídeos
video_folder = "videos"

# Pasta para salvar os vídeos processados
output_folder = "resultados"
os.makedirs(output_folder, exist_ok=True)

# Obter todos os arquivos de vídeo da pasta
supported_formats = (".mp4", ".avi", ".mov", ".mkv", ".flv")  # Formatos suportados
video_paths = [os.path.join(video_folder, f) for f in os.listdir(video_folder) if f.lower().endswith(supported_formats)]

# Verificar se há vídeos na pasta
if not video_paths:
    print("Nenhum vídeo encontrado na pasta especificada.")
else:
    for video_path in video_paths:
        print(f"Processando vídeo: {video_path}")

        # Configurar o caminho de saída
        video_name = os.path.basename(video_path)
        output_path = os.path.join(output_folder, f"processed_{video_name}")

        # Carregar o vídeo
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Não foi possível abrir o vídeo: {video_path}")
            continue

        # Obter informações do vídeo
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Inicializar o writer para salvar o vídeo processado
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        # Reinicializar o modelo para rastreamento no início de cada vídeo
        model = YOLO(model_path)  # Recarrega o modelo para reiniciar o rastreamento

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Realizar o rastreamento no frame atual
            results = model.track(frame, persist=True, conf=0.5, iou=0.5)  # Ajuste os parâmetros conforme necessário

            # Anotar as detecções no frame
            annotated_frame = frame.copy()  # Cópia do frame para anotações
            for result in results:
                if hasattr(result, "masks") and result.masks is not None:
                    masks = result.masks.data.cpu().numpy()  # Máscaras segmentadas como array numpy

                    for mask, box, confidence in zip(masks, result.boxes.xyxy, result.boxes.conf):
                        bbox = box.cpu().numpy().astype(int)
                        x_min, y_min, x_max, y_max = bbox
                        conf = confidence.item()  # Obter o valor de confiança como float

                        # Criar máscara limitada ao bbox
                        mask_resized = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Ajusta o tamanho da máscara
                        mask_resized = (mask_resized > 0.5).astype('uint8')  # Limite binário para a máscara
                        mask_roi = mask_resized[y_min:y_max, x_min:x_max]  # Limitar ao bbox

                        # Aplicar a máscara verde na área do bbox
                        green_overlay = annotated_frame[y_min:y_max, x_min:x_max].copy()
                        green_overlay[mask_roi == 1] = (0, 255, 0)  # Cor verde
                        cv2.addWeighted(green_overlay, 0.4, annotated_frame[y_min:y_max, x_min:x_max], 0.6, 0,
                                        annotated_frame[y_min:y_max, x_min:x_max])

                        # Desenhar a caixa delimitadora (bbox)
                        cv2.rectangle(annotated_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                        # Adicionar o identificador de rastreamento e confiança
                        label = f"Conf: {conf:.2f}"
                        if hasattr(result, "id") and result.id is not None:
                            tracker_id = result.id
                            label = f"ID: {tracker_id}, {label}"

                        cv2.putText(
                            annotated_frame, label,
                            (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                        )

            # Exibir o frame com as detecções
            cv2.imshow("YOLO Tracking com Rastreamento", annotated_frame)

            # Escrever o frame processado no vídeo de saída
            out.write(annotated_frame)

            # Sair com a tecla 'q'
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # Liberar recursos para o vídeo atual
        cap.release()
        out.release()
        print(f"Vídeo processado salvo em: {output_path}")

# Fechar todas as janelas abertas
cv2.destroyAllWindows()

print("Processamento finalizado para todos os vídeos.")
