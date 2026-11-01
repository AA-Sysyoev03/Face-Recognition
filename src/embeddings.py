from pathlib import Path
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import models, transforms


class ArcFaceLoss(nn.Module):
    """
    ArcFace-классификатор из третьего ноутбука.

    Сохраняем ту же структуру параметров,
    чтобы checkpoint загружался с strict=True.
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=40.0,
        m=0.57
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m

        self.weight = nn.Parameter(
            torch.FloatTensor(out_features, in_features)
        )
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, inputs, labels):
        cosine = F.linear(
            F.normalize(inputs),
            F.normalize(self.weight)
        )

        sine = torch.sqrt(
            (1.0 - cosine.pow(2)).clamp(0, 1)
        )

        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(
            cosine > self.th,
            phi,
            cosine - self.mm
        )

        one_hot = torch.zeros(
            cosine.size(),
            device=inputs.device
        )

        one_hot.scatter_(
            1,
            labels.view(-1, 1).long(),
            1
        )

        output = (
            one_hot * phi
            + (1.0 - one_hot) * cosine
        )

        output *= self.s
        return output


class FRModel_ArcFace(nn.Module):
    """
    ResNet18 + эмбеддинг размером 1024 + ArcFace.

    Архитектура соответствует модели,
    которую мы обучали в 03_face_recognition.ipynb.
    """

    def __init__(self, num_classes, embedding_size=1024):
        super().__init__()

        resnet = models.resnet18(weights=None)

        self.backbone = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        self.embedding_layer = nn.Sequential(
            nn.Linear(512, embedding_size),
            nn.BatchNorm1d(embedding_size)
        )

        self.arcface = ArcFaceLoss(
            in_features=embedding_size,
            out_features=num_classes
        )

    def forward(self, x, labels=None):
        x = self.backbone(x)
        x = torch.flatten(x, 1)

        embeddings = self.embedding_layer(x)

        if labels is not None:
            logits = self.arcface(embeddings, labels)
            return logits, embeddings

        return embeddings


# Такая же обработка, как в validation/test третьего ноутбука.
embedding_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def load_arcface(checkpoint_path, device="cpu"):
    """
    Загружает обученную модель ArcFace.

    Возвращает готовую модель в режиме eval().
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Не найден checkpoint: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model = FRModel_ArcFace(
        num_classes=checkpoint["num_classes"],
        embedding_size=checkpoint["embedding_size"]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model = model.to(device)
    model.eval()

    return model


def extract_embedding(model, aligned_face, device="cpu"):
    """
    aligned_face: PIL.Image размером 112x112.

    Возвращает нормализованный эмбеддинг
    размером (1024,) на CPU.
    """
    image_tensor = embedding_transform(
        aligned_face.convert("RGB")
    )

    image_tensor = image_tensor.unsqueeze(0).to(device)

    model.eval()

    with torch.no_grad():
        embedding = model(image_tensor)

        # Нормализация для cosine similarity
        embedding = F.normalize(embedding, p=2, dim=1)

    return embedding.squeeze(0).cpu()