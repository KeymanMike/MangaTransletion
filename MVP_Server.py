import gradio as gr
from pathlib import Path
from PIL import Image
import numpy as np

from main_app import MangaTranslationPipeline

print("Загрузка моделей...")
pipeline = MangaTranslationPipeline(
    detector_model_path='./model/manga_detector.pt',
    detector_model_size='small',
    font_path='./Font/comic.ttf',
    output_dir='Translation'
)
print("Готово!")


def process_upload(image, conf=0.25):
    if image is None:
        return None, None, "⚠️ Сначала загрузите изображение"

    tmp_path = Path("temp_upload.jpg")
    Image.fromarray(image).save(tmp_path)

    try:
        final_image = pipeline.process_image(str(tmp_path), conf=conf)

        if isinstance(final_image, (str, Path)):
            final_image = np.array(Image.open(final_image))

        tmp_path.unlink(missing_ok=True)
        return image, final_image, "Перевод выполнен успешно"

    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return image, None, f"❌ Ошибка: {str(e)}"


with gr.Blocks(title="Перевод манги") as demo:
    gr.Markdown("# 📖 Автоматический перевод манги")
    gr.Markdown("Загрузите страницу манги на **английском** языке и получите перевод на **русский**.")

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Исходная страница", type="numpy")
            conf_slider = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Порог уверенности детектора")
            btn = gr.Button("🚀 Перевести", variant="primary")

        with gr.Column():
            output_image = gr.Image(label="Переведённая страница")
            text_info = gr.Textbox(label="Распознанный текст", lines=10, max_lines=20)

    btn.click(
        fn=process_upload,
        inputs=[input_image, conf_slider],
        outputs=[input_image, output_image, text_info]
    )

    gr.Markdown("---")
    gr.Markdown("💡 **Совет:** Для лучшего качества перевода убедитесь, что страница имеет хорошее разрешение.")

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)