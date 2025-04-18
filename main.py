import os
import cv2
import torch
import numpy as np
from facenet_pytorch import MTCNN, InceptionResnetV1

# Путь к папке с изображениями известных лиц
KNOWN_FACES_DIR = 'known_faces'
# Порог для признания совпадения (чем меньше - тем строже)
THRESHOLD = 0.8

# Устройство: GPU если доступно, иначе CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Инициализация детектора и модели векторов лиц
mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Загрузка и векторизация известных лиц
known_embeddings = []
known_names = []

print("Загружаем лица из папки '{}'").format(KNOWN_FACES_DIR)
for filename in os.listdir(KNOWN_FACES_DIR):
    path = os.path.join(KNOWN_FACES_DIR, filename)
    # Имя человека по названию файла без расширения
    name, _ = os.path.splitext(filename)

    # Читаем изображение
    img = cv2.imread(path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Детектируем лицо и получаем усеченную картинку
    face = mtcnn(img_rgb)
    if face is None:
        print(f"Лицо не найдено на {filename}")
        continue

    # Вычисляем эмбеддинг
    with torch.no_grad():
        embedding = resnet(face.unsqueeze(0).to(device))
    known_embeddings.append(embedding.cpu())
    known_names.append(name)
    print(f"Загружено: {name}")

# Конвертация в один тензор
if known_embeddings:
    known_embeddings = torch.cat(known_embeddings)
else:
    print("Нет загруженных лиц! Проверьте папку.")
    exit(1)

# Запуск видеопотока
cap = cv2.VideoCapture(0)
print("Запуск видеопотока. Нажмите 'q' для выхода.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = [int(b) for b in box]
            # Кроп лица из кадра
            face_img = img_rgb[y1:y2, x1:x2]
            face_crop = cv2.resize(face_img, (160, 160))
            face_crop = torch.tensor(face_crop).permute(2, 0, 1).float()
            face_crop = (face_crop / 255.).unsqueeze(0).to(device)

            # Эмбеддинг
            with torch.no_grad():
                emb = resnet(face_crop).cpu()

            # Вычисляем расстояния до известных
            dists = (known_embeddings - emb).norm(dim=1)
            min_dist, idx = torch.min(dists, dim=0)

            # Если минимальное расстояние меньше порога - лицо распознано
            if min_dist < THRESHOLD:
                name = known_names[idx]
                label = f"{name} ({min_dist:.2f})"
            else:
                label = "Unknown"

            # Рисуем прямоугольник и подпись
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 0), 2)

    # Отображение
    cv2.imshow('Face Recognition', frame)

    # Выход по нажатию 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
