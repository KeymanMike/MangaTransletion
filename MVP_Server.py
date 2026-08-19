import gradio as gr
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd
import shutil
import tempfile
import zipfile

from main_app import MangaTranslationPipeline

_clean_image = None
_objects = None
_text_blocks = None
_translated_blocks = None

print("Загрузка моделей en→ru...")
pipeline_en_ru = MangaTranslationPipeline(
    detector_model_path='./model/manga_detector.pt',
    detector_model_size='small',
    ocr_lang='en',
    translator_src='en',
    translator_tgt='ru',
    font_path='./Font/comic.ttf',
    output_dir='Translation'
)
print("Загрузка моделей ja→ru...")
try:
    pipeline_ja_ru = MangaTranslationPipeline(
        detector_model_path='./model/manga_detector.pt',
        detector_model_size='small',
        ocr_lang='ja',
        translator_src='ja',
        translator_tgt='ru',
        font_path='./Font/comic.ttf',
        output_dir='Translation'
    )
    pipelines = {'en-ru': pipeline_en_ru, 'ja-ru': pipeline_ja_ru}
except Exception as e:
    print(f"⚠️ Японская модель не загружена: {e}")
    pipelines = {'en-ru': pipeline_en_ru}
print("Готово!")

FONT_DIR = Path("./Font")
FONT_DIR.mkdir(exist_ok=True)

def list_fonts():
    fonts = []
    for f in FONT_DIR.glob("*.ttf"):
        fonts.append(f.name)
    return sorted(fonts) if fonts else ["comic.ttf"]

font_choices = list_fonts()
print(f"Доступные шрифты: {font_choices}")


def process_single(image, conf=0.25, max_font_size=24, font_name="comic.ttf", language="en-ru"):
    if image is None:
        return None, None, "⚠️ Сначала загрузите изображение", None, gr.update(visible=False)

    pipeline = pipelines.get(language, pipeline_en_ru)
    tmp_path = Path("temp_upload.jpg")
    Image.fromarray(image).save(tmp_path)
    font_path = str(FONT_DIR / font_name) if font_name else None

    try:
        final_img, clean_img, text_blocks, translated_blocks, objects = pipeline.process_image(
            str(tmp_path), conf=conf, max_font_size=max_font_size,
            font_path=font_path, return_details=True
        )

        global _clean_image, _objects, _text_blocks, _translated_blocks
        _clean_image = clean_img
        _objects = objects
        _text_blocks = text_blocks
        _translated_blocks = translated_blocks

        # Формируем таблицу
        originals = [b.text.strip() if b.text != " " else "" for b in text_blocks]
        translations = [t.get('translated', '') for t in translated_blocks]
        df = pd.DataFrame({"Original": originals, "Translated": translations})

        if isinstance(final_img, (str, Path)):
            final_image = np.array(Image.open(final_img))
        else:
            final_image = final_img

        tmp_path.unlink(missing_ok=True)
        return image, final_image, "Перевод выполнен успешно", df, gr.update(visible=True)

    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return image, None, f"❌ Ошибка: {str(e)}", None, gr.update(visible=False)


def apply_corrections(df: pd.DataFrame, font_name: str, max_font_size: int, language: str):
    if df is None or _clean_image is None or _objects is None:
        return None, "⚠️ Нет данных для исправления"

    pipeline = pipelines.get(language, pipeline_en_ru)
    new_translations = []
    for _, row in df.iterrows():
        new_translations.append({"translated": row.get("Translated", "")})

    font_path = str(FONT_DIR / font_name) if font_name else None
    try:
        corrected_img = pipeline.apply_corrections(
            _clean_image, _objects, new_translations,
            max_font_size=max_font_size, font_path=font_path
        )
        return corrected_img, "✅ Исправления применены"
    except Exception as e:
        return None, f"❌ Ошибка: {str(e)}"

def process_folder(files, conf=0.25, max_font_size=24, font_name="comic.ttf", language="en-ru"):
    if not files:
        return None, "⚠️ Выберите изображения"

    pipeline = pipelines.get(language, pipeline_en_ru)
    font_path = str(FONT_DIR / font_name) if font_name else None
    output_tmp = tempfile.mkdtemp()
    processed_files = []

    try:
        for file in files:
            img_path = Path(file.name) if hasattr(file, 'name') else Path(file)
            final_img = pipeline.process_image(
                str(img_path), conf=conf, max_font_size=max_font_size, font_path=font_path
            )
            if isinstance(final_img, (str, Path)):
                final_img = np.array(Image.open(final_img))
            # Сохраняем в временную папку
            out_name = f"translated_{Path(img_path).stem}.png"
            out_path = Path(output_tmp) / out_name
            Image.fromarray(final_img).save(out_path)
            processed_files.append(out_path)

        if not processed_files:
            return None, "❌ Ни одно изображение не обработано"

        # Создаём zip-архив
        zip_path = Path(output_tmp) / "translated_pages.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for p in processed_files:
                zf.write(p, arcname=p.name)

        return str(zip_path), f"✅ Обработано {len(processed_files)} изображений"

    except Exception as e:
        shutil.rmtree(output_tmp, ignore_errors=True)
        return None, f"❌ Ошибка: {str(e)}"

with gr.Blocks(title="Перевод манги") as demo:
    gr.set_static_paths([])
    gr.Markdown("# 📖 Автоматический перевод манги")
    gr.Markdown("Загрузите страницу манги на **английском** языке и получите перевод на **русский**.")

    with gr.Tabs():
        # ------------------ Таб одиночного изображения ------------------
        with gr.Tab("Одиночное изображение"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(label="Исходная страница", type="numpy")
                    language_single = gr.Dropdown(
                        choices=list(pipelines.keys()), value="en-ru", label="Язык перевода"
                    )
                    conf_slider = gr.Slider(0.1, 1.0, value=0.25, step=0.05,
                                            label="Порог уверенности детектора")
                    font_dropdown = gr.Dropdown(choices=font_choices, value="comic.ttf",
                                                label="Шрифт для вставки")
                    max_font_slider = gr.Slider(12, 48, value=24, step=2,
                                                label="Макс. размер шрифта")
                    btn_single = gr.Button("🚀 Перевести", variant="primary")

                with gr.Column():
                    output_image = gr.Image(label="Переведённая страница")
                    status_single = gr.Textbox(label="Статус", value="Ожидание загрузки...", lines=2)

            gr.Markdown("### 📝 Редактирование перевода")
            with gr.Row():
                texts_table = gr.Dataframe(
                    headers=["Original", "Translated"],
                    datatype=["str", "str"],
                    interactive=True,
                    label="Распознанный и переведённый текст"
                )

            with gr.Row():
                correct_btn = gr.Button("✏️ Исправить перевод", variant="secondary")
                correction_status = gr.Textbox(label="Статус коррекции", visible=False)

            btn_single.click(
                fn=process_single,
                inputs=[input_image, conf_slider, max_font_slider, font_dropdown, language_single],
                outputs=[input_image, output_image, status_single, texts_table, correct_btn]
            )
            correct_btn.click(
                fn=apply_corrections,
                inputs=[texts_table, font_dropdown, max_font_slider, language_single],
                outputs=[output_image, correction_status]
            )

        # ------------------ Таб пакетной обработки ------------------
        with gr.Tab("Папка изображений"):
            with gr.Row():
                with gr.Column():
                    folder_input = gr.Files(label="Выберите изображения (можно несколько)", file_types=["image"])
                    language_folder = gr.Dropdown(
                        choices=list(pipelines.keys()), value="en-ru", label="Язык перевода"
                    )
                    conf_folder = gr.Slider(0.1, 1.0, value=0.25, step=0.05,
                                            label="Порог уверенности детектора")
                    font_folder = gr.Dropdown(choices=font_choices, value="comic.ttf",
                                              label="Шрифт для вставки")
                    max_font_folder = gr.Slider(12, 48, value=24, step=2,
                                                label="Макс. размер шрифта")
                    btn_folder = gr.Button("🚀 Обработать папку", variant="primary")

                with gr.Column():
                    folder_output = gr.File(label="Скачать переводы (ZIP)")
                    status_folder = gr.Textbox(label="Статус", value="Выберите файлы...", lines=2)

            btn_folder.click(
                fn=process_folder,
                inputs=[folder_input, conf_folder, max_font_folder, font_folder, language_folder],
                outputs=[folder_output, status_folder]
            )

    gr.Markdown("---")
    gr.Markdown("💡 **Совет:** Для японского языка требуется установленная библиотека manga-ocr.")
    gr.Markdown("`pip install manga-ocr`")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)