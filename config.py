"""
config.py — Конфигурация для проекта сегментации изображений
"""

# Пути к данным и результатам
DATA_IMAGES = [
    "images/image1.jpg",
    "images/image2.jpg",
    "images/image3.jpg"
]

RESULTS_DIR = "results"

# Размер изображений для обработки
TARGET_SIZE = (512, 512)

# Порог для instance segmentation (Mask R-CNN)
INSTANCE_SCORE_THRESHOLD = 0.7

# Seed для воспроизводимости
SEED = 42
