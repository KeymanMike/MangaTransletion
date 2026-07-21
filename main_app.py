import torch
import numpy as np
import cv2
from pathlib import Path
from typing import List
import matplotlib.pyplot as plt

from manga_detector import MangaBubbleDetector, MangaDataset
from manga_ocr import MangaOCR
from manga_translator import MangaTranslator
from manga_inpainter import MangaInpainter
from manga_text_inserter import MangaTextInserter

PATH_TO_IMAGE = r'./dataset/image/One Piece (41).jpg'
IMG_FOLDER = r'./dataset/image'

class MangaTranslationPipeline:
    """
    Полный пайплайн автоматического перевода манги.
    """

    def __init__(
            self,
            detector_model_path: str = None,
            detector_model_size: str = 'nano',
            ocr_lang: str = 'en',
            translator_src: str = 'en',
            translator_tgt: str = 'ru',
            font_path: str = None,
            output_dir: str = 'Translation',
            device: str = 'cuda'
    ):
        self.device = device if torch.cuda.is_available() else 'cpu'

        self.detector = MangaBubbleDetector(model_size=detector_model_size)
        if detector_model_path and Path(detector_model_path).exists():
            self.detector.load(detector_model_path)
            print(f"📦 Детектор загружен из {detector_model_path}")
        else:
            print("⚠️ Веса детектора не найдены. Сначала запустите обучение.")

        self.ocr = MangaOCR()
        self.translator = MangaTranslator(source_lang=translator_src, target_lang=translator_tgt)
        self.inpainter = MangaInpainter()
        self.inserter = MangaTextInserter(font_path=font_path, outline_width=2)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"📁 Результаты будут сохранены в {self.output_dir}")

    def train_detector(self, dataset_path: str, epochs: int = 100, **kwargs):
        """Обучает детектор на датасете."""

        dataset = MangaDataset(dataset_path, split='train')
        yaml_path = str(dataset.data_dir / 'data.yaml')

        self.detector.train(yaml_path=yaml_path, epochs=epochs, **kwargs)
        self.detector.save('manga_detector.pt')
        print("✅ Детектор обучен и сохранён в manga_detector.pt")

    def process_image(self, image_path: str, conf: float = 0.25, save_visualization: bool = False) -> np.ndarray:
        """
        Обрабатывает одно изображение и возвращает финальную картинку.
        """
        image_path = Path(image_path)
        print(f"\n{'=' * 60}")
        print(f"📄 Обработка: {image_path.name}")

        print("🔍 Детекция...")
        predictions = self.detector.predict(str(image_path), conf=conf)
        print(f"   Найдено объектов: {len(predictions['objects'])}")

        print("📖 OCR...")
        text_blocks = self.ocr.extract_all_text(predictions['image'], predictions['objects'])

        print("🌐 Перевод...")
        translated_blocks = self.translator.translate_blocks(text_blocks)

        print("🎨 Удаление текста...")
        clean_image = self.inpainter.inpaint_page(predictions['image'], predictions['objects'])

        print("📝 Вставка перевода...")
        final_image = self.inserter.insert_text(clean_image, predictions['objects'], translated_blocks)

        out_name = image_path.stem + "_translated" + image_path.suffix
        out_path = self.output_dir / out_name
        cv2.imwrite(str(out_path), cv2.cvtColor(final_image, cv2.COLOR_RGB2BGR))
        print(f"✅ Сохранено: {out_path}")

        if save_visualization:
            self._visualize(predictions['image'], final_image, image_path.name)

        return final_image

    def process_folder(self, folder_path: str, conf: float = 0.25,
                       extensions: tuple = ('.jpg', '.jpeg', '.png', '.webp')) -> List[Path]:
        """Обрабатывает все изображения в папке."""
        folder = Path(folder_path)
        image_files = []
        for ext in extensions:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))

        if not image_files:
            print(f"❌ Не найдено изображений в {folder_path}")
            return []

        print(f"📚 Найдено {len(image_files)} изображений")
        results = []
        for img_path in sorted(image_files):
            try:
                final = self.process_image(str(img_path), conf=conf)
                results.append(final)
            except Exception as e:
                print(f"❌ Ошибка при обработке {img_path.name}: {e}")
        return results

    def _visualize(self, original: np.ndarray, translated: np.ndarray, title: str):
        """Показывает сравнение оригинала и перевода."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
        ax1.imshow(original)
        ax1.set_title('Оригинал', fontsize=14)
        ax1.axis('off')
        ax2.imshow(translated)
        ax2.set_title('Перевод', fontsize=14)
        ax2.axis('off')
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    pipeline = MangaTranslationPipeline(
        detector_model_path=r'./model/manga_detector.pt',
        detector_model_size='small',
        font_path=r'./Font/comic.ttf',
        output_dir='Translation'
    )

    # Обработка одного изображения
    final = pipeline.process_image(PATH_TO_IMAGE, conf=0.25)

    # Или всей папки
    # pipeline.process_folder(IMG_FOLDER, conf=0.25)