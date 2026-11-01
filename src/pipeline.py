from pathlib import Path

import torch
from PIL import Image

from detection import FaceDetector
from hourglass import StackedHourglass
from alignment import process_face
from embeddings import load_arcface, extract_embedding


class FaceRecognitionPipeline:
    """
    Полный пайплайн распознавания лиц:

    1. Детекция всех лиц.
    2. Предсказание landmarks нашей Hourglass.
    3. Выравнивание лиц.
    4. Получение эмбеддингов нашей ArcFace.
    """

    def __init__(self, checkpoints_dir, device="cpu"):
        self.device = torch.device(device)
        checkpoints_dir = Path(checkpoints_dir)

        # 1. Готовый детектор лиц
        self.detector = FaceDetector()

        # 2. Наша обученная Stacked Hourglass
        self.hourglass = StackedHourglass(
            nstack=2,
            inp_dim=128,
            oup_dim=5
        ).to(self.device)

        hourglass_checkpoint = torch.load(
            checkpoints_dir / "best_hourglass.pth",
            map_location=self.device,
            weights_only=True
        )

        if "model_state_dict" in hourglass_checkpoint:
            hourglass_weights = hourglass_checkpoint["model_state_dict"]
        else:
            hourglass_weights = hourglass_checkpoint

        self.hourglass.load_state_dict(
            hourglass_weights,
            strict=True
        )
        self.hourglass.eval()

        # 3. Наша обученная ArcFace
        self.arcface = load_arcface(
            checkpoints_dir / "best_resnet18_arcface.pth",
            device=self.device
        )

    def recognize_faces(self, image_path):
        """
        Принимает путь к изображению.

        Возвращает список результатов для всех
        обнаруженных лиц.
        """
        image = Image.open(image_path).convert("RGB")

        # Находим все лица
        boxes = self.detector.detect(image)

        results = []

        for face_index, bbox in enumerate(boxes, start=1):
            # Hourglass + Face Alignment
            alignment_result = process_face(
                image=image,
                bbox=bbox,
                hourglass_model=self.hourglass,
                device=self.device
            )

            aligned_face = alignment_result["aligned_face"]

            # ArcFace: получаем вектор из 1024 чисел
            embedding = extract_embedding(
                model=self.arcface,
                aligned_face=aligned_face,
                device=self.device
            )

            results.append({
                "face_id": face_index,
                "bbox": alignment_result["bbox"],
                "landmarks": alignment_result["landmarks"],
                "aligned_face": aligned_face,
                "embedding": embedding
            })

        return results