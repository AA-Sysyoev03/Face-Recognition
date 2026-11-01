import cv2
import numpy as np
import torch

from PIL import Image
from torchvision import transforms


# Стандартные координаты пяти точек на изображении 112x112
REFERENCE_LANDMARKS = np.array([
    [38.2946, 51.6963],  # Левый глаз
    [73.5318, 51.5014],  # Правый глаз
    [56.0252, 71.7366],  # Нос
    [41.5493, 92.3655],  # Левый угол рта
    [70.7291, 92.2041]   # Правый угол рта
], dtype=np.float32)


def crop_face(image, bbox, margin=0.2):
    """
    Обрезает лицо с дополнительным отступом.

    image: PIL.Image
    bbox: (x1, y1, x2, y2)

    Возвращает обрезанное изображение и координаты области.
    """
    x1, y1, x2, y2 = map(float, bbox)

    width = x2 - x1
    height = y2 - y1

    x1 = max(0, int(np.floor(x1 - margin * width)))
    y1 = max(0, int(np.floor(y1 - margin * height)))

    x2 = min(image.width, int(np.ceil(x2 + margin * width)))
    y2 = min(image.height, int(np.ceil(y2 + margin * height)))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Некорректный bounding box лица")

    cropped = image.crop((x1, y1, x2, y2))

    return cropped, (x1, y1, x2, y2)


def predict_landmarks(model, cropped_face, device):
    """
    Предсказывает 5 ключевых точек лица.

    Координаты возвращаются относительно cropped_face.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    tensor = transform(cropped_face.convert("RGB"))
    tensor = tensor.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        output = model(tensor)

    # Берём heatmaps последнего стека Hourglass
    heatmaps = output[0, -1].detach().cpu().numpy()

    points = []

    for heatmap in heatmaps:
        y, x = np.unravel_index(
            np.argmax(heatmap),
            heatmap.shape
        )

        # Переводим координаты из heatmap 64x64
        # обратно в размер исходного cropped_face.
        point_x = float(x) * cropped_face.width / heatmap.shape[1]
        point_y = float(y) * cropped_face.height / heatmap.shape[0]

        points.append([point_x, point_y])

    return np.asarray(points, dtype=np.float32)


def align_face(cropped_face, landmarks):
    """
    Выравнивает лицо до 112x112 по пяти точкам.
    """
    matrix, inliers = cv2.estimateAffinePartial2D(
        landmarks,
        REFERENCE_LANDMARKS,
        method=cv2.LMEDS
    )

    if matrix is None:
        raise RuntimeError(
            "Не удалось вычислить преобразование для выравнивания"
        )

    image_np = np.asarray(cropped_face.convert("RGB"))

    aligned = cv2.warpAffine(
        image_np,
        matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT
    )

    return Image.fromarray(aligned)


def process_face(image, bbox, hourglass_model, device):
    """
    Полная обработка одного найденного лица.
    """
    cropped_face, crop_box = crop_face(
        image,
        bbox,
        margin=0.2
    )

    landmarks_local = predict_landmarks(
        hourglass_model,
        cropped_face,
        device
    )

    aligned_face = align_face(
        cropped_face,
        landmarks_local
    )

    # Переводим локальные точки в координаты всей фотографии
    landmarks_global = landmarks_local.copy()
    landmarks_global[:, 0] += crop_box[0]
    landmarks_global[:, 1] += crop_box[1]

    return {
        "bbox": tuple(bbox),
        "crop_box": crop_box,
        "landmarks": landmarks_global,
        "aligned_face": aligned_face
    }