import cv2

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
from PIL import Image, ImageDraw, ImageFont

from manga_detector import MangaBubbleDetector
from manga_ocr import MangaOCR
from manga_inpainter import MangaInpainter
from manga_translator import MangaTranslator


PATH_TO_IMAGE=r'./dataset/image/One Piece (41).jpg'


class MangaTextInserter:
    """
    Вставляет переведённый текст в пузыри манги.
    Бинарный поиск шрифта, круглая обводка, кэш шрифтов, компактные холсты.
    """

    def __init__(self, font_path: str = None, outline_width: int = 2,
                 max_font_size: int = 24, min_font_size: int = 6,
                 inner_margin: float = 0.10):
        self.font_path = font_path
        self.outline_width = outline_width
        self.max_font_size = max_font_size
        self.min_font_size = min_font_size
        self.inner_margin = inner_margin
        self._font_cache = {}
        if font_path:
            try:
                _ = ImageFont.truetype(font_path, size=24)
            except:
                print(f"⚠️ Шрифт {font_path} не найден.")
                self.font_path = None

    def _get_font(self, size: int) -> ImageFont.FreeTypeFont:
        if size not in self._font_cache:
            if self.font_path:
                self._font_cache[size] = ImageFont.truetype(self.font_path, size=size)
            else:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def _get_text_color(self, bubble_img: np.ndarray, mask: np.ndarray) -> Tuple[
        Tuple[int, int, int], Tuple[int, int, int]]:
        if mask is None or mask.sum() == 0:
            return (0, 0, 0), (255, 255, 255)
        bg = bubble_img[mask > 0]
        if len(bg) == 0:
            return (0, 0, 0), (255, 255, 255)
        gray = cv2.cvtColor(bg.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).ravel()
        med = np.median(gray)
        if med > 128:
            return (0, 0, 0), (255, 255, 255)
        else:
            return (255, 255, 255), (0, 0, 0)

    def _wrap_text(self, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
        words = text.split()
        if not words:
            return []
        lines = []
        cur_line = []
        cur_w = 0
        space_w = font.getbbox(' ')[2] - font.getbbox(' ')[0]
        for w in words:
            bbox = font.getbbox(w)
            w_w = bbox[2] - bbox[0]
            add = space_w + w_w if cur_line else w_w
            if cur_w + add <= max_width:
                cur_line.append(w)
                cur_w += add
            else:
                if cur_line:
                    lines.append(' '.join(cur_line))
                cur_line = [w]
                cur_w = w_w
        if cur_line:
            lines.append(' '.join(cur_line))
        return lines

    def _get_line_height(self, font: ImageFont.FreeTypeFont) -> int:
        bbox = font.getbbox('Ag')
        return bbox[3] - bbox[1] + 4

    def _fit_text(self, text: str, avail_w: int, avail_h: int) -> Tuple[int, List[str]]:
        """
        Бинарный поиск максимального размера шрифта, при котором текст
        умещается в avail_w x avail_h.
        """
        lo, hi = self.min_font_size, self.max_font_size
        best_size = self.min_font_size
        best_lines = []

        font = self._get_font(hi)
        lines = self._wrap_text(text, font, avail_w)
        if self._get_line_height(font) * len(lines) <= avail_h:
            return hi, lines

        while lo <= hi:
            mid = (lo + hi) // 2
            font = self._get_font(mid)
            lines = self._wrap_text(text, font, avail_w)
            line_h = self._get_line_height(font)
            if line_h * len(lines) <= avail_h:
                best_size = mid
                best_lines = lines
                lo = mid + 1
            else:
                hi = mid - 1

        return best_size - 1, best_lines

    def insert_text(self, image: np.ndarray, blocks: List[dict], translated_blocks: List[dict]) -> np.ndarray:
        result = image.copy()
        h_img, w_img = result.shape[:2]

        for block, trans in zip(blocks, translated_blocks):
            text = trans.get('translated', '')
            if not text or not text.strip():
                continue

            mask_full = block.get('mask')
            bbox = block.get('bbox')
            if mask_full is None or mask_full.sum() < 10 or bbox is None:
                continue

            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)

            bubble_img = result[y1:y2, x1:x2]
            bubble_mask = mask_full[y1:y2, x1:x2].copy()
            bubble_h, bubble_w = bubble_img.shape[:2]

            ys, xs = np.where(bubble_mask > 0)
            if len(xs) == 0:
                continue
            mx1, mx2 = xs.min(), xs.max()
            my1, my2 = ys.min(), ys.max()
            mask_w = mx2 - mx1
            mask_h = my2 - my1
            margin_w = int(mask_w * self.inner_margin)
            margin_h = int(mask_h * self.inner_margin)
            inset_x1 = max(mx1 + margin_w, 0)
            inset_y1 = max(my1 + margin_h, 0)
            inset_x2 = min(mx2 - margin_w, bubble_w)
            inset_y2 = min(my2 - margin_h, bubble_h)
            avail_w = max(1, inset_x2 - inset_x1)
            avail_h = max(1, inset_y2 - inset_y1)

            font_size, lines = self._fit_text(text, avail_w, avail_h)
            if not lines:
                continue
            font = self._get_font(font_size)
            line_h = self._get_line_height(font)
            total_h = line_h * len(lines)

            text_color, outline_color = self._get_text_color(bubble_img, bubble_mask)

            text_img = Image.new('RGBA', (bubble_w, bubble_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(text_img)

            y_start = inset_y1 + (avail_h - total_h) // 2

            r = self.outline_width
            r2 = r * r

            for line in lines:
                line_bbox = font.getbbox(line)
                line_w = line_bbox[2] - line_bbox[0]
                x_line = inset_x1 + (avail_w - line_w) // 2

                if self.outline_width > 0:
                    for dx in range(-r, r + 1):
                        for dy in range(-r, r + 1):
                            if dx == 0 and dy == 0:
                                continue
                            if dx * dx + dy * dy <= r2:
                                draw.text((x_line + dx, y_start + dy), line,
                                          fill=outline_color + (255,), font=font)
                draw.text((x_line, y_start), line, fill=text_color + (255,), font=font)
                y_start += line_h

            text_arr = np.array(text_img)
            alpha = text_arr[:, :, 3]
            alpha = cv2.bitwise_and(alpha, alpha, mask=bubble_mask.astype(np.uint8))
            text_arr[:, :, 3] = alpha

            alpha_factor = alpha.astype(float)[:, :, np.newaxis] / 255.0
            blended = (bubble_img * (1 - alpha_factor) + text_arr[:, :, :3] * alpha_factor).astype(np.uint8)
            result[y1:y2, x1:x2] = blended

        return result

    def debug_insert(self, image: np.ndarray, block: dict, translated_text: str):
        mask_full = block.get('mask')
        bbox = block.get('bbox')
        if mask_full is None or bbox is None:
            print("Нет маски или bbox")
            return

        x1, y1, x2, y2 = map(int, bbox)
        roi = image[y1:y2, x1:x2].copy()
        roi_mask = mask_full[y1:y2, x1:x2]

        temp_image = image.copy()
        fake_blocks = [block]
        fake_trans = [{'translated': translated_text}]
        result_block = self.insert_text(temp_image, fake_blocks, fake_trans)

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(roi)
        axes[0].set_title('Оригинал')
        axes[0].axis('off')

        mask_vis = np.zeros_like(roi)
        mask_vis[roi_mask > 0] = 255
        axes[1].imshow(mask_vis, cmap='gray')
        axes[1].set_title('Маска')
        axes[1].axis('off')

        result_roi = result_block[y1:y2, x1:x2]
        axes[2].imshow(result_roi)
        axes[2].set_title('После вставки')
        axes[2].axis('off')
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    detector = MangaBubbleDetector(model_size='small')
    detector.load(r'./model/manga_detector.pt')
    predictions = detector.predict(PATH_TO_IMAGE, conf=0.25)

    ocr = MangaOCR()
    translator = MangaTranslator(source_lang='en', target_lang='ru')
    text_blocks = ocr.extract_all_text(predictions['image'], predictions['objects'])
    translated_blocks = translator.translate_blocks(text_blocks)

    inpainter = MangaInpainter()
    clean_image = inpainter.inpaint_page(predictions['image'], predictions['objects'])
    inserter = MangaTextInserter(font_path='./Font/comic.ttf', outline_width=2)
    final_image = inserter.insert_text(clean_image, predictions['objects'], translated_blocks)

    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(predictions['image'])
    plt.title('Оригинал')
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(final_image)
    plt.title('После inpainting')
    plt.axis('off')
    plt.show()