
import cv2

from pathlib import Path
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

import easyocr

from manga_detector import MangaBubbleDetector


@dataclass
class TextBlock:
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]


class MangaOCR:
    """
    OCR для манги на EasyOCR.
    Пробует 2 варианта: оригинал и CLAHE + ресайз 300.
    Не очищает текст – чистка перенесена в переводчик.
    Нераспознанные блоки возвращаются с text=" ".
    """

    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        print("🔤 EasyOCR готов")

    def prepare_region(
        self, image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox)

        crop = image[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2]

        result = np.ones_like(crop) * 255
        result[crop_mask > 0] = crop[crop_mask > 0]

        gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)

        text_mean = cv2.mean(gray, mask=crop_mask.astype(np.uint8))[0]
        bg_mean = cv2.mean(gray, mask=(1 - crop_mask).astype(np.uint8))[0]

        if text_mean > bg_mean:
            gray = 255 - gray

        return gray

    def _to_rgb(self, gray: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if gray.ndim == 2 else gray

    def _ocr(self, image: np.ndarray) -> Tuple[str, float]:
        results = self.reader.readtext(
            image,
            paragraph=False,
            min_size=5,
            text_threshold=0.4,
            low_text=0.2,
            width_ths=0.5,
        )
        if results:
            texts, confs = [], []
            for _, txt, conf in results:
                txt = txt.strip()
                if txt and conf > 0.2:
                    texts.append(txt)
                    confs.append(conf)
            if texts:
                return ' '.join(texts), np.mean(confs)
        return "", 0.0

    def recognize(self, gray: np.ndarray) -> Tuple[str, float]:
        best_text, best_conf = "", 0.0

        h, w = gray.shape
        scale = 300 / max(h, w) if max(h, w) < 300 else 1.0

        # Вариант 1: оригинал
        v1 = (
            cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            if scale > 1.0
            else gray
        )
        text, conf = self._ocr(self._to_rgb(v1))
        if conf > best_conf:
            best_text, best_conf = text, conf

        # Вариант 2: CLAHE + ресайз 300
        v2 = self.clahe.apply(gray)
        if scale > 1.0:
            v2 = cv2.resize(v2, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        text, conf = self._ocr(self._to_rgb(v2))
        if conf > best_conf:
            best_text, best_conf = text, conf

        return best_text, best_conf

    def recognize_text(
        self, image: np.ndarray, mask: np.ndarray, bbox: Tuple[int, int, int, int]
    ) -> TextBlock:
        """Всегда возвращает TextBlock. Если текст пуст или слишком короток – text=' '."""
        try:
            gray = self.prepare_region(image, mask, bbox)
            text, conf = self.recognize(gray)
        except Exception:
            text, conf = "", 0.0

        if not text or len(text.strip()) < 2:
            return TextBlock(text=" ", confidence=conf, bbox=bbox)

        return TextBlock(text=text.strip(), confidence=conf, bbox=bbox)

    def extract_all_text(self, image: np.ndarray, objects: list) -> List[TextBlock]:
        blocks = []
        for i, obj in enumerate(objects):
            mask = obj.get('mask')
            bbox = obj.get('bbox')
            if mask is None or mask.sum() < 10:
                blocks.append(TextBlock(text=" ", confidence=0.0, bbox=tuple(bbox) if bbox else (0, 0, 0, 0)))
                print(f"   [{i+1}/{len(objects)}] empty mask → ' '")
                continue

            block = self.recognize_text(image, mask, tuple(bbox))
            status = f'"{block.text[:60]}" ({block.confidence:.2f})' if block.text.strip() != "" else "' ' (empty)"
            print(f"   [{i+1}/{len(objects)}] {status}")
            blocks.append(block)

        n_text = sum(1 for b in blocks if b.text.strip() and b.text != " ")
        print(f"   {n_text}/{len(blocks)} blocks with text")
        return blocks


def validate_pipeline(detector, image_path, ocr, conf=0.25):
    predictions = detector.predict(image_path, conf=conf)
    image = predictions['image']
    objects = predictions['objects']
    n = len(objects)

    print(f"\n{'=' * 60}")
    print(f"📷 {Path(image_path).name} | областей: {n}")
    print(f"{'=' * 60}\n")

    vis = image.copy()
    for i, obj in enumerate(objects):
        x1, y1, x2, y2 = map(int, obj['bbox'])
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(vis, f"{i + 1}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    plt.figure(figsize=(14, 10))
    plt.imshow(vis)
    plt.title(f'Детекция: {n} областей', fontsize=14)
    plt.axis('off')
    plt.show()

    for i, obj in enumerate(objects):
        mask = obj['mask']
        bbox = obj['bbox']
        x1, y1, x2, y2 = map(int, bbox)

        crop = image[y1:y2, x1:x2].copy()
        crop_mask = mask[y1:y2, x1:x2]

        masked = np.ones_like(crop) * 255
        masked[crop_mask > 0] = crop[crop_mask > 0]

        easy_input = ocr.prepare_region(image, mask, tuple(bbox))
        block = ocr.recognize_text(image, mask, tuple(bbox))

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(crop)
        axes[0].set_title(f'Оригинал\n{crop.shape[:2]}', fontsize=9)
        axes[0].axis('off')

        axes[1].imshow(masked)
        axes[1].set_title('Маска + белый фон', fontsize=9)
        axes[1].axis('off')

        axes[2].imshow(easy_input)
        axes[2].set_title(f'EasyOCR вход\n{easy_input.shape[:2]}', fontsize=9)
        axes[2].axis('off')

        result = block.text if block else '❌'
        plt.suptitle(f'Блок {i + 1} ({obj["class_name"]}): "{result}"', fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.show()

        print(f"   [{i + 1}] \"{result}\"")

    blocks = ocr.extract_all_text(image, objects)
    print(f"\n📊 {len(blocks)}/{n} распознано")
    return blocks

if __name__ == "__main__":
    detector = MangaBubbleDetector(model_size='small')
    detector.load(r'./model/manga_detector.pt')
    ocr = MangaOCR()
    blocks = validate_pipeline(detector, r'./dataset/image/One Piece (41).jpg', ocr)