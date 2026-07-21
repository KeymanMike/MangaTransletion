# Автоматический перевод манги с английского на русский (не финал).  
Colab:  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1KW5X4WAyQtOXAkccVQ6UJ4lgM7vMB2l5?usp=sharing) 

Проект выполняет полный цикл: детекция пузырей, OCR, перевод, удаление оригинального текста и вставка перевода.

[MANGA Translation ex.webm](https://github.com/user-attachments/assets/264c6213-f08d-45a1-a95d-ba125f6d9dc9)

---

## ✨ Возможности

- **Детекция** текстовых элементов (пузыри, SFX, фоновые надписи) с помощью YOLOv8-seg
- **Распознавание** английского текста (EasyOCR с адаптивной предобработкой)
- **Перевод** на русский язык (NLLB‑200 1.3B или OPUS‑MT)
- **Удаление** оригинального текста (inpainting с точной маской)
- **Вставка** перевода с сохранением читаемости (кэш шрифтов, круговая обводка, бинарный поиск размера)
- **Веб‑интерфейс** на Gradio для загрузки страниц и просмотра результатов

---
## 🚀 Быстрый старт (веб‑интерфейс)
  ```bash
  python MVP_Server.py
```
---
## 🐍 Использование через Python
  ```python
from main_app import MangaTranslationPipeline

pipeline = MangaTranslationPipeline(
    detector_model_path='./model/manga_detector.pt',
    detector_model_size='small',
    font_path='./Font/comic.ttf',
    output_dir='Translation'
)
final_image = pipeline.process_image('page.jpg', conf=0.25)
```
---
## 📦 Зависимости

См. requirements.txt. Основные библиотеки:
   - torch, ultralytics – детекция (YOLOv8-seg)
   - easyocr – распознавание текста
   - transformers, sentencepiece – перевод (NLLB-200)
   - opencv-python-headless, Pillow – обработка изображений
   - gradio – веб‑интерфейс
   - simple-lama-inpainting (опционально) – улучшенное удаление текста

---
## 🤝 Благодарности
 * Ultralytics
 * YOLO
 * EasyOCR
 * Meta NLLB-200
 * LaMa Inpainting
 * Gradio
