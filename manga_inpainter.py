import cv2
import numpy as np
import matplotlib.pyplot as plt

from manga_detector import MangaBubbleDetector



class MangaInpainter:
    """
    Удаляет текст, используя точную маску пузыря.
    """
    def __init__(self, white_threshold=240, variance_threshold=50.0,
                 edge_safe_margin=2, text_dilate=2):
        self.white_threshold = white_threshold
        self.variance_threshold = variance_threshold
        self.edge_safe_margin = edge_safe_margin   # на сколько пикселей сужаем маску пузыря
        self.text_dilate = text_dilate             # на сколько расширяем маску текста

    def _get_bubble_background(self, image: np.ndarray, bubble_mask: np.ndarray) -> np.ndarray:
        """Доминирующий цвет фона пузыря (медиана)."""
        inside = image[bubble_mask > 0]
        if len(inside) == 0:
            return np.array([255, 255, 255], dtype=np.uint8)
        gray = cv2.cvtColor(inside.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).flatten()
        threshold = np.percentile(gray, 20) if len(gray) > 20 else 0
        bright_mask = gray >= threshold
        if np.sum(bright_mask) < 10:
            return np.median(inside, axis=0).astype(np.uint8)
        return np.median(inside[bright_mask], axis=0).astype(np.uint8)

    def _is_uniform(self, image: np.ndarray, mask: np.ndarray) -> bool:
        """Проверяет однородность фона внутри маски."""
        pixels = image[mask > 0]
        if len(pixels) < 10:
            return True
        return np.std(pixels) < self.variance_threshold

    def inpaint_page(self, image: np.ndarray, blocks: list) -> np.ndarray:
        result = image.copy()
        h, w = result.shape[:2]
        kernel_erode = np.ones((3, 3), np.uint8)
        kernel_dilate = np.ones((3, 3), np.uint8)

        for block in blocks:
            mask_full = block.get('mask')
            bbox = block.get('bbox')
            class_name = block.get('class_name', 'bubble')

            if mask_full is None or mask_full.sum() < 10:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            roi = result[y1:y2, x1:x2]
            roi_mask = mask_full[y1:y2, x1:x2].copy()

            roi_mask_safe = cv2.erode(roi_mask, kernel_erode, iterations=self.edge_safe_margin)

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            _, binary_text = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            text_mask = cv2.bitwise_and(binary_text, binary_text, mask=roi_mask_safe.astype(np.uint8))
            text_mask = cv2.dilate(text_mask, kernel_dilate, iterations=self.text_dilate)
            text_mask = cv2.bitwise_and(text_mask, text_mask, mask=roi_mask_safe.astype(np.uint8))

            if class_name in ('bubble', 'narrative_box') and self._is_uniform(roi, roi_mask):
                bg_color = self._get_bubble_background(roi, roi_mask)
                roi[text_mask > 0] = bg_color
                result[y1:y2, x1:x2] = roi
            else:
                if text_mask.sum() < 10:
                    continue
                roi_inpainted = cv2.inpaint(roi, text_mask, 3, cv2.INPAINT_TELEA)

                result_roi = roi.copy()
                result_roi[roi_mask_safe > 0] = roi_inpainted[roi_mask_safe > 0]
                result[y1:y2, x1:x2] = result_roi

        return result

if __name__ == '__main__':
    detector = MangaBubbleDetector(model_size='small')
    detector.load(r'./model/manga_detector.pt')
    predictions = detector.predict(r'./dataset/image/One Piece (41).jpg', conf=0.25)

    inpainter = MangaInpainter()
    clean = inpainter.inpaint_page(predictions['image'], predictions['objects'])

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(predictions['image'])
    plt.title('Оригинал')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(clean)
    plt.title('После inpainting')
    plt.axis('off')
    plt.show()