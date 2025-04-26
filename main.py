"""
main.py — Запуск и визуализация сегментации изображений
- Semantic segmentation: DeepLabV3
- Instance segmentation: Mask R-CNN
- Конфигурируется через config.py
- Модули: segmentation.py, utils.py
"""

import os
import matplotlib.pyplot as plt
import logging
from config import DATA_IMAGES, RESULTS_DIR, SEED
from segmentation import semantic_segmentation, instance_segmentation
from utils import set_seed, setup_logging

# --- Настройка среды ---
set_seed(SEED)
setup_logging()

def process_image(image_path, output_dir=RESULTS_DIR):
    """
    Обрабатывает одно изображение: строит и сохраняет сравнение оригинала, семантической и инстанс-сегментации.
    """
    os.makedirs(output_dir, exist_ok=True)
    try:
        img_resized, semantic_result = semantic_segmentation(image_path)
        _, instance_result = instance_segmentation(image_path)

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.title('Оригинал')
        plt.imshow(img_resized)
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.title('Семантическая сегментация')
        plt.imshow(semantic_result)
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.title('Инстанс-сегментация')
        plt.imshow(instance_result)
        plt.axis('off')
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{base_name}_results.png")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Результат сохранён: {output_path}")
        return output_path
    except Exception as e:
        logging.error(f"Ошибка при обработке {image_path}: {e}")
        return None


def main():
    """
    Основной запуск: обрабатывает список изображений из config.py
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    logging.info("=== Начало обработки изображений ===")
    for img_path in DATA_IMAGES:
        process_image(img_path, RESULTS_DIR)
    logging.info("✅ Обработка завершена!")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        process_image(image_path, RESULTS_DIR)
    else:
        main()