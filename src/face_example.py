from pathlib import Path
from io import BytesIO
import zipfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent

ARCHIVE_PATH = ROOT / "data" / "aligned_faces_112.zip"
CSV_PATH = ROOT / "data" / "processed" / "splits" / "test.csv"
EMBEDDINGS_PATH = (
    ROOT / "results" / "embedding_evaluation" / "test_embeddings.npz"
)
OUTPUT_DIR = ROOT / "results" / "embedding_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

THRESHOLD = 0.0984


def main():
    # Загружаем информацию о фотографиях
    df = pd.read_csv(CSV_PATH).reset_index(drop=True)

    # Загружаем уже вычисленные эмбеддинги
    saved = np.load(EMBEDDINGS_PATH)
    embeddings = saved["embeddings"]
    labels = saved["identity_ids"].astype(str)

    if len(df) != len(embeddings):
        raise ValueError(
            "Количество строк CSV не совпадает с числом эмбеддингов."
        )

    # Проверяем соответствие порядка личностей
    if not np.array_equal(df["identity_id"].astype(str).to_numpy(), labels):
        raise ValueError(
            "Порядок identity_id в CSV и эмбеддингах не совпадает."
        )

    # Группируем фотографии по identity_id
    groups = {}

    for idx, identity in enumerate(labels):
        groups.setdefault(identity, []).append(idx)

    # Находим первую личность минимум с двумя фотографиями
    same_indices = next(
        indices[:2]
        for indices in groups.values()
        if len(indices) >= 2
    )

    a1, a2 = same_indices

    # Выбираем фотографию другой личности
    other_identity = next(
        identity
        for identity in groups
        if identity != labels[a1]
    )

    b1 = groups[other_identity][0]

    # Сходство нормализованных эмбеддингов
    sim_same = float(np.dot(embeddings[a1], embeddings[a2]))
    sim_diff = float(np.dot(embeddings[a1], embeddings[b1]))

    print(f"Один человек: {sim_same:.4f}")
    print(f"Разные люди: {sim_diff:.4f}")
    print(f"Порог: {THRESHOLD:.4f}")

    def read_face(zf, index):
        image_id = Path(str(df.iloc[index]["image_id"])).name
        archive_name = f"images/{image_id}"

        with zf.open(archive_name) as file:
            return Image.open(BytesIO(file.read())).convert("RGB")

    with zipfile.ZipFile(ARCHIVE_PATH) as zf:
        face_a1 = read_face(zf, a1)
        face_a2 = read_face(zf, a2)
        face_b1 = read_face(zf, b1)

    # Рисуем две строки: один человек и разные люди
    fig, axes = plt.subplots(2, 2, figsize=(7, 7))

    axes[0, 0].imshow(face_a1)
    axes[0, 1].imshow(face_a2)

    axes[1, 0].imshow(face_a1)
    axes[1, 1].imshow(face_b1)

    for ax in axes.flat:
        ax.axis("off")

    axes[0, 0].set_title("Person A — photo 1")
    axes[0, 1].set_title("Person A — photo 2")
    axes[1, 0].set_title("Person A")
    axes[1, 1].set_title("Person B")

    fig.text(
        0.5, 0.95,
        f"Same identity | cosine similarity = {sim_same:.4f}",
        ha="center", fontsize=12, weight="bold"
    )

    fig.text(
        0.5, 0.49,
        f"Different identities | cosine similarity = {sim_diff:.4f}",
        ha="center", fontsize=12, weight="bold"
    )

    fig.suptitle(
        f"ArcFace verification examples | threshold = {THRESHOLD:.4f}",
        y=1.02, fontsize=13
    )

    plt.subplots_adjust(
        top=0.88, bottom=0.04, hspace=0.65, wspace=0.1
    )

    output_path = OUTPUT_DIR / "face_comparison_examples.png"

    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close()

    print(f"\nИллюстрация сохранена: {output_path}")


if __name__ == "__main__":
    main()