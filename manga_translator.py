import re
from pathlib import Path

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from manga_detector import MangaBubbleDetector
from manga_ocr import MangaOCR


class MangaTranslator:
    """
    Переводчик EN→RU через NLLB-200-1.3B.
    Пакетный перевод с точечной предобработкой:
    1. Удаление артефактов (^, ~, _ и т.д.)
    2. Удаление заиканий (B-But → But)
    3. Склеивание переносов (REMEM- BER → REMEMBER)
    4. Нормализация регистра (Title Case)
    5. Исправление конечной пунктуации
    Каждый текст разбивается на предложения, переводится отдельно и собирается обратно.
    """

    def __init__(self, source_lang: str = 'en', target_lang: str = 'ru'):
        self.source_lang = source_lang
        self.target_lang = target_lang

        model_name = "facebook/nllb-200-1.3B"
        print(f"🌐 NLLB-200-1.3B: {source_lang} → {target_lang}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("   ✅ GPU (fp16)")
        else:
            print("   ✅ CPU (fp32)")

        self.src_code = "eng_Latn"
        self.tgt_code = "rus_Cyrl"

    def _get_tgt_token_id(self) -> int:
        tid = self.tokenizer.convert_tokens_to_ids(self.tgt_code)
        if tid != self.tokenizer.unk_token_id:
            return tid
        for token, token_id in self.tokenizer.get_added_vocab().items():
            if token == self.tgt_code:
                return token_id
        return self.tokenizer.unk_token_id

    def _remove_artifacts(self, text: str) -> str:
        allowed = re.compile(r'[^a-zA-Z0-9\s.,!?;:\'\"\-\(\)…–—]')
        return allowed.sub('', text)

    def _fix_stuttering(self, text: str) -> str:
        def _stutter_repl(m: re.Match) -> str:
            prefix = m.group(1)
            word = m.group(2)
            if len(prefix) <= 2 and word.lower().startswith(prefix.lower()):
                return word
            return m.group(0)
        return re.sub(r'\b([A-Za-z]{1,2})-([A-Za-z]{3,})', _stutter_repl, text)

    def _fix_hyphenation(self, text: str) -> str:
        return re.sub(r'(\w)-\s+(\w)', r'\1\2', text)

    def _normalize_case(self, text: str) -> str:
        if not text:
            return text
        sentences = re.split(r'(?<=[.?!])\s+', text)
        normalized = []
        for sent in sentences:
            if not sent:
                continue
            words = sent.split()
            if words:
                words[0] = words[0].capitalize()
                words = [w if w == 'I' else w.lower() for w in words]
                normalized.append(' '.join(words))
        return ' '.join(normalized)

    def _fix_ending_punctuation(self, text: str) -> str:
        if not text:
            return text
        last_punct_match = re.search(r'[^\w\s]+$', text)
        if last_punct_match:
            punct = last_punct_match.group()
            if all(c in '.?!' for c in punct):
                return text
            else:
                return re.sub(r'[^\w\s]+$', '.', text)
        else:
            return text + '.'

    def _preprocess(self, text: str) -> str:
        text = self._remove_artifacts(text)
        text = self._fix_stuttering(text)
        text = self._fix_hyphenation(text)
        text = self._normalize_case(text)
        text = self._fix_ending_punctuation(text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


    def _split_into_sentences(self, text: str) -> list[str]:
        """Разбивает текст на предложения по .?! и возвращает список строк."""
        sentences = re.split(r'(?<=[.?!])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _postprocess_translation(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([«„(])\s+', r'\1', text)
        text = re.sub(r'\s+([»“)])', r'\1', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def translate_batch(self, texts: list[str]) -> list[str]:
        all_sentences = []
        sentence_map = []

        for text in texts:
            if not text or not text.strip():
                sentence_map.append([])
                continue
            text = self._preprocess(text)
            sents = self._split_into_sentences(text)
            start_idx = len(all_sentences)
            all_sentences.extend(sents)
            sentence_map.append(list(range(start_idx, len(all_sentences))))

        if not all_sentences:
            return [""] * len(texts)

        self.tokenizer.src_lang = self.src_code
        inputs = self.tokenizer(
            all_sentences,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=True
        )

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        forced_bos = self._get_tgt_token_id()

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=forced_bos,
                max_length=256,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=2,
            )

        decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translated_sents = [self._postprocess_translation(t.strip()) for t in decoded]

        final_translations = []
        for indices in sentence_map:
            if not indices:
                final_translations.append("")
            else:
                trans_parts = [translated_sents[idx] for idx in indices]
                final_translations.append(' '.join(trans_parts))
        return final_translations

    def translate_blocks(self, text_blocks: list) -> list:
        print(f"\n🌐 Перевод {len(text_blocks)} блоков...")

        texts = []
        for block in text_blocks:
            t = block.text if hasattr(block, 'text') else block.get('text', '')
            texts.append(t.strip() if t else "")

        translated_texts = self.translate_batch(texts)

        results = []
        for i, (block, trans) in enumerate(zip(text_blocks, translated_texts)):
            orig = texts[i] if texts[i] else " "
            results.append({
                'original': orig,
                'translated': trans,
                'confidence': getattr(block, 'confidence', 0.0),
                'bbox': getattr(block, 'bbox', None)
            })
            if i < 10 and orig.strip():
                print(f"   {orig[:60]} → {trans[:60]}")

        print(f"   ✅ {len(results)} блоков")
        return results


def process_page(detector, ocr, translator, image_path, conf=0.25):
    """
    Полный пайплайн: YOLO → OCR → перевод.
    """
    print(f"\n{'=' * 60}")
    print(f"📄 {Path(image_path).name}")
    print(f"{'=' * 60}")

    print("\n🔍 Детекция...")
    predictions = detector.predict(image_path, conf=conf)
    print(f"   Найдено: {len(predictions['objects'])} областей")

    print("\n📖 OCR...")
    text_blocks = ocr.extract_all_text(predictions['image'], predictions['objects'])

    translated = translator.translate_blocks(text_blocks)

    print(f"\n{'=' * 60}")
    print("📋 РЕЗУЛЬТАТЫ")
    print(f"{'=' * 60}")
    for i, item in enumerate(translated):
        print(f"\n   [{i + 1}] {item['original']}")
        print(f"       → {item['translated']}")

    return translated

if __name__ == "__main__":
    detector = MangaBubbleDetector(model_size='small')
    detector.load(r'./model/manga_detector.pt')
    ocr = MangaOCR()
    translator = MangaTranslator(source_lang='en', target_lang='ru')

    results = process_page(detector, ocr, translator, r'./dataset/image/One Piece (41).jpg')