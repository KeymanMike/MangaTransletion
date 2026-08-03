import gradio as gr
from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd


from main_app import MangaTranslationPipeline

_clean_image = None
_objects = None
_text_blocks = None
_translated_blocks = None

print("Загрузка моделей...")
pipeline = MangaTranslationPipeline(
    detector_model_path='./model/manga_detector.pt',
    detector_model_size='small',
    font_path='./Font/comic.ttf',
    output_dir='Translation'
)
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


def process_upload(image, conf=0.25, max_font_slider=24, font_name="comic.ttf"):
    if image is None:
        return None, None, "⚠️ Сначала загрузите изображение"

    tmp_path = Path("temp_upload.jpg")
    Image.fromarray(image).save(tmp_path)
    font_path = str(FONT_DIR / font_name) if font_name else None

    try:
        final_img, clean_img, text_blocks, translated_blocks, object = pipeline.process_image(
            str(tmp_path), conf=conf, max_font_size=max_font_slider,
            font_path=font_path, return_details=True
        )

        global _clean_image, _objects, _translated_blocks
        _clean_image = clean_img
        _text_blocks = text_blocks
        _objects = object
        _translated_blocks = translated_blocks

        originals = [b.text.strip() if b.text != " " else "" for b in text_blocks]
        translations = [t.get('translated', '') for t in translated_blocks]
        df = pd.DataFrame({"Original": originals, "Translated": translations})

        if isinstance(final_img, (str, Path)):
            final_image = np.array(Image.open(final_img))

        tmp_path.unlink(missing_ok=True)
        return image, final_img, "Перевод выполнен успешно", df, gr.update(visible=True)

    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return image, None, f"❌ Ошибка: {str(e)}"

def apply_corrections(df: pd.DataFrame, font_name: str, max_font_size: int):
    if df is None or _clean_image is None or _objects is None:
        return None, "⚠️ Нет данных для исправления"

    # Извлекаем исправленные переводы из таблицы
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


with gr.Blocks(title="Перевод манги") as demo:
    gr.set_static_paths([])
    gr.Markdown("# 📖 Автоматический перевод манги")
    gr.Markdown("Загрузите страницу манги на **английском** языке и получите перевод на **русский**.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Исходная страница", type="numpy")
            conf_slider = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Порог уверенности детектора")
            font_dropdown = gr.Dropdown(choices=font_choices, value="comic.ttf", label="Шрифт для вставки")
            max_font_slider = gr.Slider(12, 48, value=24, step=2, label="Макс. размер шрифта")
            btn = gr.Button("🚀 Перевести", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Переведённая страница")
            status = gr.Textbox(label="Статус", value="Ожидание загрузки...", lines=2)

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

    btn.click(
        fn=process_upload,
        inputs=[input_image, conf_slider, max_font_slider,font_dropdown],
        outputs=[input_image, output_image, status, texts_table, correct_btn]
    )
    correct_btn.click(
        fn=apply_corrections,
        inputs=[texts_table, font_dropdown, max_font_slider],
        outputs=[output_image, correction_status]
    )

    gr.Markdown("---")
    gr.Markdown("💡 **Совет:** Для лучшего качества перевода убедитесь, что страница имеет хорошее разрешение.")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)