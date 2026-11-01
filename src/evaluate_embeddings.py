from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from torchvision import transforms

from embeddings import load_arcface, embedding_transform


ROOT = Path(__file__).resolve().parent.parent

ARCHIVE = ROOT / "data" / "aligned_faces_112.zip"
CSV_DIR = ROOT / "data" / "processed" / "splits"
CHECKPOINT = ROOT / "checkpoints" / "best_resnet18_arcface.pth"
OUTPUT_DIR = ROOT / "results" / "embedding_evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32 if DEVICE.type == "cuda" else 8
SEED = 42

np.random.seed(SEED)
torch.manual_seed(SEED)


class AlignedCelebADataset(Dataset):
    def __init__(self, dataframe, archive_path):
        self.df = dataframe.reset_index(drop=True)
        self.archive_path = archive_path
        self.archive = None

        with zipfile.ZipFile(self.archive_path) as zf:
            self.archive_names = set(zf.namelist())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.archive is None:
            self.archive = zipfile.ZipFile(self.archive_path)

        row = self.df.iloc[index]
        filename = Path(str(row["image_id"])).name
        archive_name = f"images/{filename}"

        if archive_name not in self.archive_names:
            raise FileNotFoundError(
                f"В архиве отсутствует {archive_name}"
            )

        import io

        with self.archive.open(archive_name) as file:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")

        tensor = embedding_transform(image)
        identity = str(row["identity_id"])

        return tensor, identity


def extract_all_embeddings(model, dataframe, name):
    dataset = AlignedCelebADataset(dataframe, ARCHIVE)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    embeddings = []
    identities = []

    print(f"\nОбработка {name}: {len(dataset)} изображений", flush=True)

    with torch.inference_mode():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(DEVICE)

            batch_embeddings = model(images)
            batch_embeddings = F.normalize(
                batch_embeddings, p=2, dim=1
            )

            embeddings.append(batch_embeddings.cpu())
            identities.extend(labels)

            if (batch_idx + 1) % 20 == 0:
                processed = min(
                    (batch_idx + 1) * BATCH_SIZE,
                    len(dataset)
                )
                print(
                    f"  {processed}/{len(dataset)}",
                    flush=True
                )

    embeddings = torch.cat(embeddings).numpy()

    np.savez_compressed(
        OUTPUT_DIR / f"{name}_embeddings.npz",
        embeddings=embeddings,
        identity_ids=np.array(identities)
    )

    return embeddings, np.array(identities)


def build_pairs(embeddings, labels, max_positive=5000, max_negative=5000):
    """
    Формируем пары одного и разных людей.
    """
    rng = np.random.default_rng(SEED)

    groups = {}
    for idx, identity in enumerate(labels):
        groups.setdefault(identity, []).append(idx)

    positive_pairs = []

    for indices in groups.values():
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                positive_pairs.append((indices[i], indices[j]))

    if not positive_pairs:
        raise ValueError("Нет положительных пар для сравнения.")

    rng.shuffle(positive_pairs)
    positive_pairs = positive_pairs[:max_positive]

    negative_pairs = []
    seen = set()

    while len(negative_pairs) < max_negative:
        i, j = rng.choice(len(labels), size=2, replace=False)

        if labels[i] == labels[j]:
            continue

        pair = (min(i, j), max(i, j))

        if pair in seen:
            continue

        seen.add(pair)
        negative_pairs.append(pair)

    pairs = positive_pairs + negative_pairs

    similarities = np.array([
        np.dot(embeddings[i], embeddings[j])
        for i, j in pairs
    ])

    targets = np.array(
        [1] * len(positive_pairs)
        + [0] * len(negative_pairs)
    )

    return similarities, targets


def evaluate_split(embeddings, labels, name, threshold=None):
    similarities, targets = build_pairs(embeddings, labels)

    auc = roc_auc_score(targets, similarities)

    fpr, tpr, thresholds = roc_curve(
        targets, similarities
    )

    if threshold is None:
        # Порог подбирается только на validation.
        best_idx = np.argmax(tpr - fpr)
        threshold = float(thresholds[best_idx])

    predictions = (similarities >= threshold).astype(int)
    accuracy = float((predictions == targets).mean())

    print(f"\n{name.upper()}")
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Pair accuracy: {accuracy:.4f}")
    print(f"Threshold: {threshold:.4f}")

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ArcFace ROC — {name}")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_roc.png", dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(
        similarities[targets == 1],
        bins=40,
        alpha=0.6,
        label="Same identity"
    )
    plt.hist(
        similarities[targets == 0],
        bins=40,
        alpha=0.6,
        label="Different identities"
    )
    plt.axvline(
        threshold, linestyle="--", label="Threshold"
    )
    plt.xlabel("Cosine similarity")
    plt.ylabel("Number of pairs")
    plt.title(f"ArcFace similarity distribution — {name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"{name}_similarities.png",
        dpi=160
    )
    plt.close()

    return threshold, auc, accuracy


def main():
    print("Устройство:", DEVICE, flush=True)

    for path in [ARCHIVE, CHECKPOINT,
                 CSV_DIR / "val.csv", CSV_DIR / "test.csv"]:
        if not path.exists():
            raise FileNotFoundError(f"Не найден файл: {path}")

    val_df = pd.read_csv(CSV_DIR / "val.csv")
    test_df = pd.read_csv(CSV_DIR / "test.csv")

    for df, split in [(val_df, "val"), (test_df, "test")]:
        needed = {"image_id", "identity_id"}
        if not needed.issubset(df.columns):
            raise ValueError(
                f"В {split}.csv нужны столбцы {needed}. "
                f"Найдены: {list(df.columns)}"
            )

    val_ids = set(val_df["identity_id"])
    test_ids = set(test_df["identity_id"])

    if val_ids & test_ids:
        raise ValueError(
            "Validation и test содержат одинаковые личности!"
        )

    print("Validation identities:", len(val_ids))
    print("Test identities:", len(test_ids))

    print("Загружаем ArcFace...", flush=True)
    model = load_arcface(CHECKPOINT, device=DEVICE)

    val_emb, val_labels = extract_all_embeddings(
        model, val_df, "val"
    )

    test_emb, test_labels = extract_all_embeddings(
        model, test_df, "test"
    )

    threshold, val_auc, val_acc = evaluate_split(
        val_emb, val_labels, "val"
    )

    _, test_auc, test_acc = evaluate_split(
        test_emb, test_labels, "test",
        threshold=threshold
    )

    summary = (
        f"Validation ROC-AUC: {val_auc:.4f}\n"
        f"Validation pair accuracy: {val_acc:.4f}\n"
        f"Test ROC-AUC: {test_auc:.4f}\n"
        f"Test pair accuracy: {test_acc:.4f}\n"
        f"Threshold (selected on validation): {threshold:.4f}\n"
    )

    (OUTPUT_DIR / "metrics.txt").write_text(
        summary, encoding="utf-8"
    )

    print("\n===== ИТОГ =====")
    print(summary)
    print("Результаты:", OUTPUT_DIR)


if __name__ == "__main__":
    main()