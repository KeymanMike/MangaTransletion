import os
import json
import gc
import cv2
import random

from pathlib import Path
import shutil
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from typing import Dict, List, Optional



import torch
from torch.utils.data import Dataset

from ultralytics import YOLO

def clear_memory():

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    print("memory cleared")

def coco_to_yolo_seg(coco_json, img_dir, output_dir, label_map={'bubble': 0, 'sfx_text': 1}):
    with open(coco_json) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    images = {img['id']: img for img in data['images']}

    anns_by_image = {}
    for ann in data['annotations']:
        if not ann.get('segmentation'):
            continue
        img_id = ann['image_id']
        anns_by_image.setdefault(img_id, []).append(ann)

    for img_id, anns in anns_by_image.items():
        img_info = images[img_id]
        w, h = img_info['width'], img_info['height']
        filename = img_info['file_name']
        stem = Path(filename).stem

        yolo_lines = []
        for ann in anns:
            cat_id = ann['category_id']
            cat_name = None
            for cat in data['categories']:
                if cat['id'] == cat_id:
                    cat_name = cat['name']
                    break
            if cat_name not in label_map:
                continue
            class_idx = label_map[cat_name]

            segs = ann['segmentation']
            for seg in segs:
                if len(seg) < 6:
                    continue
                norm_seg = []
                points = zip(seg[0::2], seg[1::2])
                for x, y in points:
                    norm_seg.append(x / w)
                    norm_seg.append(y / h)
                yolo_lines.append(f"{class_idx} " + " ".join(f"{c:.6f}" for c in norm_seg))

        if yolo_lines:
            txt_path = os.path.join(output_dir, f"{stem}.txt")
            with open(txt_path, 'w') as f:
                f.write("\n".join(yolo_lines))
    print("Конвертация завершена!")


class MangaDataset(Dataset):
    """
    PyTorch Dataset для загрузки изображений манги с полигональными аннотациями.

    При инициализации создаёт структуру датасета в data/ с разбиением на train/val/test
    и файлом data.yaml.
    """

    def __init__(
            self,
            dataset_path: str,
            split: str = 'all',
            train_ratio: float = 0.7,
            val_ratio: float = 0.15,
            seed: int = 42
    ):
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / 'image'
        self.labels_dir = self.dataset_path / 'yolo_labels'
        self.split = split

        assert self.images_dir.exists(), f"Нет папки: {self.images_dir}"
        assert self.labels_dir.exists(), f"Нет папки: {self.labels_dir}"

        self.classes = {
            0: 'bubble',
            1: 'sfx_text',
            2: 'background_text'
        }
        self.num_classes = len(self.classes)

        self.class_colors = {
            0: (0.2, 0.4, 1.0),
            1: (1.0, 0.2, 0.2),
            2: (0.2, 0.8, 0.2)
        }

        self.samples = self._collect_samples()
        print(f"📦 Найдено {len(self.samples)} изображений с аннотациями")

        if split != 'all':
            self.samples = self._split_samples(train_ratio, val_ratio, seed)
            print(f"   Режим '{split}': {len(self.samples)} изображений")

        self.data_dir = self.dataset_path.parent / 'data'
        self._build_dataset(train_ratio, val_ratio, seed)

    def _collect_samples(self) -> List[Dict[str, Path]]:
        """Находит все пары изображение-аннотация"""
        samples = []
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp', '*.bmp']:
            image_files.extend(self.images_dir.glob(ext))

        for img_path in sorted(image_files):
            label_path = self.labels_dir / (img_path.stem + '.txt')
            if label_path.exists():
                with open(label_path, 'r') as f:
                    if f.read().strip():
                        samples.append({'image': img_path, 'label': label_path})
                    else:
                        print(f"⚠️  Пустая аннотация: {label_path.name}")
            else:
                print(f"⚠️  Нет аннотации: {img_path.name}")

        return samples

    def _split_samples(self, train_ratio: float, val_ratio: float, seed: int) -> List[Dict]:
        """Разбивает на train/val/test"""
        random.seed(seed)
        indices = list(range(len(self.samples)))
        random.shuffle(indices)

        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if self.split == 'train':
            selected = indices[:n_train]
        elif self.split == 'val':
            selected = indices[n_train:n_train + n_val]
        elif self.split == 'test':
            selected = indices[n_train + n_val:]
        else:
            raise ValueError(f"Неизвестный split: {self.split}")

        return [self.samples[i] for i in selected]

    def _build_dataset(self, train_ratio: float, val_ratio: float, seed: int):
        """
        Создаёт структуру датасета:
        data/
        ├── train/
        │   ├── images/
        │   └── labels/
        ├── val/
        │   ├── images/
        │   └── labels/
        ├── test/
        │   ├── images/
        │   └── labels/
        └── data.yaml
        """
        if self.data_dir.exists():
            shutil.rmtree(self.data_dir)

        random.seed(seed)
        indices = list(range(len(self._collect_samples())))
        random.shuffle(indices)
        all_samples = self._collect_samples()

        n = len(all_samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        splits = {
            'train': [all_samples[i] for i in indices[:n_train]],
            'val': [all_samples[i] for i in indices[n_train:n_train + n_val]],
            'test': [all_samples[i] for i in indices[n_train + n_val:]]
        }

        for split_name, split_samples in splits.items():
            img_dir = self.data_dir / split_name / 'images'
            lbl_dir = self.data_dir / split_name / 'labels'
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)

            for sample in split_samples:
                safe_name = sample['image'].stem
                for char in [' ', '(', ')', '[', ']', '&', "'", '"']:
                    safe_name = safe_name.replace(char, '_')

                shutil.copy2(sample['image'], img_dir / f"{safe_name}{sample['image'].suffix}")
                shutil.copy2(sample['label'], lbl_dir / f"{safe_name}.txt")

        print(str(self.data_dir.absolute()))
        config = {
            'path': str(self.data_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': self.classes
        }

        yaml_path = self.data_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

        print(f"✅ Датасет создан: {self.data_dir}")
        print(f"   train: {len(splits['train'])} | val: {len(splits['val'])} | test: {len(splits['test'])}")
        print(f"   Конфиг: {yaml_path}")

    def _parse_yolo_annotation(self, label_path: Path, w: int, h: int) -> Dict:
        """Парсит YOLO-аннотацию в абсолютные координаты"""
        polygons, labels, boxes = [], [], []

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:
                    continue

                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]

                abs_points = []
                for i in range(0, len(coords), 2):
                    abs_points.append([coords[i] * w, coords[i + 1] * h])

                abs_points = np.array(abs_points, dtype=np.float32)

                polygons.append(abs_points)
                labels.append(class_id)
                boxes.append([
                    abs_points[:, 0].min(), abs_points[:, 1].min(),
                    abs_points[:, 0].max(), abs_points[:, 1].max()
                ])

        masks = np.zeros((len(polygons), h, w), dtype=np.uint8)
        for i, poly in enumerate(polygons):
            cv2.fillPoly(masks[i], [poly.astype(np.int32)], 1)

        return {
            'polygons': polygons,
            'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4)),
            'labels': np.array(labels, dtype=np.int64),
            'masks': masks
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        image = cv2.imread(str(sample['image']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        target = self._parse_yolo_annotation(sample['label'], w, h)

        return {
            'image': image,
            'target': target,
            'image_path': str(sample['image']),
            'size': (h, w)
        }

    def visualize(self, idx: Optional[int] = None, image_path: Optional[str] = None, save_path: Optional[str] = None):
        """
        Визуализирует один сэмпл.

        Аргументы:
            idx: индекс сэмпла (если None — случайный)
            image_path: путь к конкретному изображению
            save_path: сохранить визуализацию
        """
        if image_path:
            found = False
            for i, s in enumerate(self.samples):
                if s['image'].name == Path(image_path).name:
                    idx = i
                    found = True
                    break
            if not found:
                raise ValueError(f"Изображение не найдено: {image_path}")

        if idx is None:
            idx = random.randint(0, len(self) - 1)

        sample = self[idx]
        image = sample['image'].copy()
        target = sample['target']
        h, w = sample['size']

        fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        ax.imshow(image)

        for i, poly in enumerate(target['polygons']):
            class_id = target['labels'][i]
            color = self.class_colors.get(class_id, (1, 1, 1))
            name = self.classes[class_id]

            patch = Polygon(poly, closed=True, facecolor=color, edgecolor=color, alpha=0.3, linewidth=2)
            ax.add_patch(patch)

            center = poly.mean(axis=0)
            ax.text(center[0], center[1], name, ha='center', va='center',
                    color='white', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.7))

        unique, counts = np.unique(target['labels'], return_counts=True)
        stats = "\n".join([f"  {self.classes[c]}: {cnt}" for c, cnt in zip(unique, counts)])

        ax.set_title(f"{Path(sample['image_path']).name} | {w}×{h}\n{stats}", fontsize=11)
        ax.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ {save_path}")

        plt.show()

        print(f"\n📷 {Path(sample['image_path']).name}")
        print(f"   Размер: {w}×{h} | Объектов: {len(target['polygons'])}")
        for c, cnt in zip(unique, counts):
            print(f"   {self.classes[c]}: {cnt}")

    def print_statistics(self):
        """Выводит статистику датасета"""
        distribution = {name: 0 for name in self.classes.values()}
        for s in self.samples:
            target = self._parse_yolo_annotation(s['label'], 1, 1)
            for l in target['labels']:
                distribution[self.classes[l]] += 1

        total = sum(distribution.values())

        print(f"\n📦 СТАТИСТИКА")
        print(f"   Режим: {self.split}")
        print(f"   Изображений: {len(self.samples)}")
        print(f"   Объектов: {total}")
        for name, count in distribution.items():
            print(f"   {name}: {count} ({count / total * 100:.1f}%)" if total else f"   {name}: 0")
        print(f"   Среднее объектов/изобр: {total / len(self.samples):.1f}" if self.samples else "")


class MangaBubbleDetector:
    """
    Детектор текстовых элементов манги на основе YOLOv8-seg.

    Использует готовую структуру датасета от MangaDataset.
    Основной метод — predict_single() для одного изображения,
    результат сразу готов для OCR и перевода.
    """

    def __init__(self, model_size: str = 'nano'):
        model_map = {
            'nano': 'yolov8n-seg.pt',
            'small': 'yolov8s-seg.pt',
            'medium': 'yolov8m-seg.pt',
        }

        self.model_name = model_map.get(model_size, 'yolov8n-seg.pt')
        self.model_size = model_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO(self.model_name)

        self.class_names = {
            0: 'bubble',
            1: 'sfx_text',
            2: 'background_text'
        }

        self.colors = {
            0: (255, 50, 50),
            1: (50, 50, 255),
            2: (50, 255, 50)
        }

        print(f"🚀 Детектор YOLOv8-{model_size}-seg | {self.device}")

    def train(
            self,
            yaml_path: str,
            epochs: int = 100,
            imgsz: int = 640,
            batch: int = 8,
            use_augmentation: bool = False,
            **kwargs
    ):
        """
        Обучает модель, используя готовый data.yaml от MangaDataset.

        Args:
            yaml_path: путь к data.yaml
            epochs: количество эпох
            imgsz: размер изображений
            batch: размер батча
            use_augmentation: включить лёгкие аугментации
        """
        print(f"\n🎯 Обучение: {epochs} эпох, batch={batch}, imgsz={imgsz}")
        print(f"   Аугментации: {'вкл' if use_augmentation else 'выкл'}")
        print(yaml_path)

        train_args = {
            'data': yaml_path,
            'epochs': epochs,
            'imgsz': imgsz,
            'batch': batch,
            'device': self.device,
            'lr0': 0.001,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'optimizer': 'SGD',
            'cos_lr': True,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'overlap_mask': True,
            'mask_ratio': 4,
            'save': True,
            'save_period': 10,
            'patience': 50,
            'pretrained': True,
            'seed': 42,
            'project': 'manga_detector',
            'name': f'manga_{self.model_size}',
            'exist_ok': True,
            'verbose': True,
            'plots': True,
        }

        if use_augmentation:
            train_args.update({
                'hsv_h': 0.01, 'hsv_s': 0.3, 'hsv_v': 0.2,
                'degrees': 5.0, 'translate': 0.05, 'scale': 0.3, 'shear': 1.0,
                'flipud': 0.0, 'fliplr': 0.3,
                'mosaic': 0.3, 'mixup': 0.05, 'copy_paste': 0.05,
                'close_mosaic': 5,
            })
        else:
            train_args.update({
                'hsv_h': 0.0, 'hsv_s': 0.0, 'hsv_v': 0.0,
                'degrees': 0.0, 'translate': 0.0, 'scale': 0.0, 'shear': 0.0,
                'flipud': 0.0, 'fliplr': 0.0,
                'mosaic': 0.0, 'mixup': 0.0, 'copy_paste': 0.0,
            })

        train_args.update(kwargs)

        self.model.train(**train_args)

        best_path = Path('manga_detector') / f'manga_{self.model_size}' / 'weights' / 'best.pt'
        if best_path.exists():
            self.model = YOLO(str(best_path))
            print(f"✅ Лучшая модель: {best_path}")

        print("✅ Обучение завершено")

    def validate(self, yaml_path: str) -> Dict:
        """
        Валидация модели.

        Args:
            yaml_path: путь к data.yaml

        Returns:
            словарь с метриками
        """
        print(f"\n📊 Валидация: {yaml_path}")

        metrics = self.model.val(data=yaml_path, split='val', device=self.device)

        results = {}
        if hasattr(metrics, 'box') and metrics.box is not None:
            results['box_p'] = float(metrics.box.p)
            results['box_r'] = float(metrics.box.r)
            results['box_map50'] = float(metrics.box.map50)
            results['box_map'] = float(metrics.box.map)
            print(f"   Box: P={results['box_p']:.3f} R={results['box_r']:.3f} "
                  f"mAP50={results['box_map50']:.3f} mAP={results['box_map']:.3f}")

        if hasattr(metrics, 'seg') and metrics.seg is not None:
            results['mask_p'] = float(metrics.seg.p)
            results['mask_r'] = float(metrics.seg.r)
            results['mask_map50'] = float(metrics.seg.map50)
            results['mask_map'] = float(metrics.seg.map)
            print(f"   Mask: P={results['mask_p']:.3f} R={results['mask_r']:.3f} "
                  f"mAP50={results['mask_map50']:.3f} mAP={results['mask_map']:.3f}")

        return results

    def predict(self, image_path: str, conf: float = 0.25) -> Dict:
        """
        Предсказание на одном изображении.
        Возвращает структуру, готовую для OCR.

        Args:
            image_path: путь к изображению
            conf: порог уверенности

        Returns:
            {
                'image': np.ndarray (RGB),
                'size': (w, h),
                'objects': [
                    {
                        'class_id': int,
                        'class_name': str,
                        'confidence': float,
                        'bbox': [x1, y1, x2, y2],
                        'mask': np.ndarray (бинарная, h×w),
                    },
                    ...
                ]
            }
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить: {image_path}")

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        results = self.model.predict(source=image_path, conf=conf, device=self.device, verbose=False)
        result = results[0]

        objects = []

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()

            for i in range(len(masks)):
                mask = masks[i]

                cls_val = result.boxes.cls[i]
                conf_val = result.boxes.conf[i]
                xyxy = result.boxes.xyxy[i]

                class_id = int(cls_val.item()) if hasattr(cls_val, 'item') else int(cls_val)
                confidence = float(conf_val.item()) if hasattr(conf_val, 'item') else float(conf_val)
                bbox = [float(x.item()) if hasattr(x, 'item') else float(x) for x in xyxy]

                mask_resized = cv2.resize(mask, (w, h))
                mask_binary = (mask_resized > 0.5).astype(np.uint8)

                objects.append({
                    'class_id': class_id,
                    'class_name': self.class_names.get(class_id, 'unknown'),
                    'confidence': confidence,
                    'bbox': bbox,
                    'mask': mask_binary,
                })

        objects.sort(key=lambda o: (o['bbox'][1], o['bbox'][0]))

        return {
            'image': image_rgb,
            'size': (w, h),
            'objects': objects,
        }

    def draw_predictions(self, predictions: Dict) -> np.ndarray:
        """Рисует предсказания на изображении."""
        vis = predictions['image'].copy()

        for obj in predictions['objects']:
            color = self.colors.get(obj['class_id'], (255, 255, 0))

            mask = obj['mask']
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                overlay = vis.copy()
                cv2.fillPoly(overlay, [cnt], color)
                cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
                cv2.polylines(vis, [cnt], True, color, 2)

            x1, y1, x2, y2 = map(int, obj['bbox'])
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

            label = f"{obj['class_name']} {obj['confidence']:.2f}"
            cv2.putText(vis, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return vis

    def compare_with_gt(self, image_path: str, label_path: str, conf: float = 0.25):
        """Сравнивает предсказания с ground truth (визуализация)."""

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        preds = self.predict(image_path, conf=conf)
        pred_vis = self.draw_predictions(preds)

        gt_vis = image.copy()
        gt_count = 0
        if Path(label_path).exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 7:
                        class_id = int(parts[0])
                        coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)
                        coords[:, 0] *= w
                        coords[:, 1] *= h
                        coords = coords.astype(np.int32)

                        color = self.colors.get(class_id, (255, 255, 0))
                        overlay = gt_vis.copy()
                        cv2.fillPoly(overlay, [coords], color)
                        cv2.addWeighted(overlay, 0.3, gt_vis, 0.7, 0, gt_vis)
                        cv2.polylines(gt_vis, [coords], True, color, 2)
                        gt_count += 1

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].imshow(image);
        axes[0].set_title('Оригинал');
        axes[0].axis('off')
        axes[1].imshow(gt_vis);
        axes[1].set_title(f'GT ({gt_count})');
        axes[1].axis('off')
        axes[2].imshow(pred_vis);
        axes[2].set_title(f'Pred ({len(preds["objects"])})');
        axes[2].axis('off')
        plt.tight_layout()
        plt.show()

        return preds

    def save(self, path: str = 'manga_detector.pt'):
        folder_path = Path(path).parent
        folder_path.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str):
        p = Path(path)
        if p.exists():
            self.model = YOLO(str(p))
            print(f"📂 Модель загружена: {p}")
        else:
            raise FileNotFoundError(f"Файл модели не найден: {path}")


if __name__ == "__main__":
        coco_to_yolo_seg(coco_json='./dataset/annotations/instances_default.json',
                         img_dir='image',
                         output_dir='yolo_labels'
                        )

        dataset = MangaDataset(
            dataset_path='./dataset',
            split='train',
            train_ratio=0.7,
            val_ratio=0.15,
            seed=42
        )

        # dataset.print_statistics()
        # dataset.visualize()
        # dataset.visualize(image_path='/content/image/One Piece (41).jpg')

        detector = MangaBubbleDetector(model_size='small')
        detector.train(yaml_path=str(dataset.data_dir / 'data.yaml'), epochs=100)

        # Валидируем
        detector.validate(yaml_path=str(dataset.data_dir / 'data.yaml'))

        # Тестируем на одиночном изображении
        test_img = r'./dataset/image/One Piece (41).jpg'
        test_lbl = r'./dataset/yolo_labels/One Piece (41).txt'

        pred = detector.compare_with_gt(
            image_path=test_img,
            label_path=test_lbl
        )

        detector.save(r'./model/manga_detector.pt')