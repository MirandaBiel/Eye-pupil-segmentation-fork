import os
from ultralytics import YOLO
import cv2

# Carregar o modelo treinado
model = YOLO("pupil_tracker.pt")

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
    # Processar cada vídeo
    for video_path in video_paths:
        print(f"Processando vídeo: {video_path}")
        
        try:
            # Obter o nome do arquivo para salvar o vídeo processado
            video_name = os.path.basename(video_path)
            output_path = os.path.join(output_folder, f"processed_{video_name}")
            
            # Abrir o vídeo original
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Erro ao abrir o vídeo: {video_path}")
                continue

            # Obter informações do vídeo
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Processar os quadros usando o modelo
            results = model.track(
                source=video_path,
                conf=0.5,
                iou=0.5,
                tracker="botsort.yaml",
                stream=True,  # Usar stream para evitar sobrecarga de memória
            )

            for result in results:
                frame = result.orig_img  # Quadro original
                boxes = result.boxes  # Boxes para os objetos detectados
                masks = result.masks  # Máscaras de segmentação (se disponíveis)
                probs = result.probs  # Probabilidades das classes (se disponíveis)

                # Adicionar anotações no quadro
                if boxes:
                    boxes.plot(frame)  # Desenhar as caixas delimitadoras no quadro
                if masks:
                    masks.plot(frame)  # Desenhar as máscaras no quadro
                
                # Salvar o quadro processado no vídeo de saída
                out.write(frame)

            # Liberar recursos
            cap.release()
            out.release()
            print(f"Vídeo processado salvo em: {output_path}")
        
        except Exception as e:
            print(f"Erro ao processar o vídeo {video_path}: {e}")

    print("Processamento finalizado para todos os vídeos.")
