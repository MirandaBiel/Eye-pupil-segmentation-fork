import os
from ultralytics import YOLO

# Carregar o modelo treinado
model = YOLO("pupil_tracker.pt")

# Pasta contendo os vídeos
video_folder = "videos"

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
            # Executar o rastreamento no vídeo atual com rastreador especificado
            results = model.track(
                source=video_path,      # Caminho para o vídeo
                conf=0.5,               # Limite de confiança (ajuste conforme necessário)
                iou=0.5,                # Limite de IOU (ajuste conforme necessário)
                tracker="bytetrack.yaml", # Use "bytetrack.yaml" ou "botsort.yaml" conforme necessário
                show=False,             # Não mostrar o vídeo em tempo real
                save=True,              # Salvar o vídeo anotado no diretório padrão de saída
            )
            print(f"Rastreamento concluído para: {video_path}")
        except Exception as e:
            print(f"Erro ao processar o vídeo {video_path}: {e}")

    print("Processamento finalizado para todos os vídeos.")
