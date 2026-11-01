from pathlib import Path

import cv2
import numpy as np
from PIL import Image


class FaceDetector:
    def __init__(self):
        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.detector = cv2.CascadeClassifier(cascade_path)

        if self.detector.empty():
            raise RuntimeError("Не удалось загрузить детектор лиц")

    def detect(self, image):
        """
        image: PIL.Image в RGB

        Возвращает список bounding boxes:
        [(x1, y1, x2, y2), ...]
        """
        image_rgb = np.asarray(image.convert("RGB"))

        gray = cv2.cvtColor(
            image_rgb,
            cv2.COLOR_RGB2GRAY
        )

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        boxes = []

        for x, y, w, h in faces:
            boxes.append((
                int(x),
                int(y),
                int(x + w),
                int(y + h)
            ))

        return boxes