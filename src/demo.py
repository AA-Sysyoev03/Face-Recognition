import argparse
from pathlib import Path

import torch
from PIL import Image, ImageDraw

from pipeline import FaceRecognitionPipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"


def main():
    parser = argparse.ArgumentParser(
        description="Face Recognition: Detection + Hourglass + ArcFace"
    )

    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Путь к фотографии"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "pipeline_results",
        help="Папка для результатов"
    )

    args = parser.parse_args()

    if not args.image.is_file():
        raise FileNotFoundError(f"Фотография не найдена: {args.image}")

    args.output.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Устройство: {device}")
    print("Загрузка обученных моделей...", flush=True)

    pipeline = FaceRecognitionPipeline(
        checkpoints_dir=CHECKPOINTS_DIR,
        device=device
    )

    print("Обработка фотографии...", flush=True)

    results = pipeline.recognize_faces(args.image)

    original = Image.open(args.image).convert("RGB")
    annotated = original.copy()
    draw = ImageDraw.Draw(annotated)

    print(f"Найдено лиц: {len(results)}")

    for result in results:
        face_id = result["face_id"]
        bbox = result["bbox"]
        landmarks = result["landmarks"]
        embedding = result["embedding"]

        # Bounding box
        draw.rectangle(bbox, outline="lime", width=3)

        # Номер лица
        draw.text(
            (bbox[0], max(0, bbox[1] - 15)),
            f"Face {face_id}",
            fill="lime"
        )

        # Пять ключевых точек
        for x, y in landmarks:
            r = 3
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill="red"
            )

        # Выровненное лицо
        result["aligned_face"].save(
            args.output / f"face_{face_id}_aligned.jpg"
        )

        # Эмбеддинг из 1024 чисел
        torch.save(
            embedding,
            args.output / f"face_{face_id}_embedding.pt"
        )

        print(
            f"Лицо {face_id}: bbox={bbox}, "
            f"embedding={tuple(embedding.shape)}"
        )

    annotated.save(args.output / "detected_faces.jpg")

    print(f"\nРезультаты сохранены в: {args.output}")


if __name__ == "__main__":
    main()