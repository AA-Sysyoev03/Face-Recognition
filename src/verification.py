import torch
import torch.nn.functional as F


def cosine_similarity(embedding1, embedding2):
    """
    Вычисляет косинусное сходство двух эмбеддингов.

    embedding1, embedding2: тензоры размером (1024,)

    Возвращает число от -1 до 1.
    """
    embedding1 = embedding1.float().flatten()
    embedding2 = embedding2.float().flatten()

    if embedding1.shape != embedding2.shape:
        raise ValueError(
            "Эмбеддинги должны иметь одинаковый размер"
        )

    similarity = F.cosine_similarity(
        embedding1.unsqueeze(0),
        embedding2.unsqueeze(0),
        dim=1
    )

    return similarity.item()