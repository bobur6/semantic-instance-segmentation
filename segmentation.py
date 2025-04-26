"""
segmentation.py — Модули для семантической и инстанс-сегментации
"""
import torch
import torchvision
import numpy as np
import cv2
from torchvision import transforms
from config import TARGET_SIZE, INSTANCE_SCORE_THRESHOLD


def resize_with_padding(img, target_size):
    """
    Масштабирует изображение с сохранением пропорций и паддингом до target_size (tuple)
    Возвращает padded_img, параметры crop (x_offset, y_offset, new_w, new_h)
    """
    h, w = img.shape[:2]
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    pad_w, pad_h = target_size[0] - new_w, target_size[1] - new_h
    x_offset, y_offset = pad_w // 2, pad_h // 2
    padded = np.zeros((target_size[1], target_size[0], 3), dtype=img.dtype)
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return padded, (x_offset, y_offset, new_w, new_h), (h, w)


def crop_to_original(img, crop_params, orig_shape):
    x_offset, y_offset, new_w, new_h = crop_params
    h, w = orig_shape
    cropped = img[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_NEAREST)


def semantic_segmentation(image_path, target_size=TARGET_SIZE):
    """Семантическая сегментация с сохранением пропорций"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DEFAULT')
    model.eval().to(device)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    padded_img, crop_params, orig_shape = resize_with_padding(img, target_size)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(padded_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)['out'][0]
    mask = output.argmax(0).cpu().numpy()
    semantic_mask = (mask > 0).astype(np.uint8)
    colored = np.zeros_like(padded_img)
    colored[semantic_mask == 1] = [255, 255, 0]
    colored[semantic_mask == 0] = [0, 0, 128]
    # Обрезаем к оригинальному размеру
    colored_cropped = crop_to_original(colored, crop_params, orig_shape)
    img_cropped = crop_to_original(padded_img, crop_params, orig_shape)
    return img_cropped, colored_cropped


def instance_segmentation(image_path, target_size=TARGET_SIZE, score_thresh=INSTANCE_SCORE_THRESHOLD):
    """Инстанс-сегментация с сохранением пропорций"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights='DEFAULT')
    model.eval().to(device)

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    padded_img, crop_params, orig_shape = resize_with_padding(img, target_size)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    img_tensor = transform(padded_img).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    masks = predictions[0]['masks'].cpu().numpy()
    scores = predictions[0]['scores'].cpu().numpy()
    high_conf_indices = np.where(scores > score_thresh)[0]

    instance_result = padded_img.copy()
    colors = [
        [230, 180, 80], [180, 120, 240], [150, 230, 230], [240, 130, 40],
        [255, 100, 100], [100, 255, 100], [100, 100, 255]
    ]
    for i, idx in enumerate(high_conf_indices):
        mask = masks[idx][0] > 0.5
        color = np.array(colors[i % len(colors)])
        colored_mask = np.zeros_like(instance_result)
        colored_mask[mask] = color
        instance_result = cv2.addWeighted(instance_result, 0.7, colored_mask, 0.3, 0)
    # Обрезаем к оригинальному размеру
    instance_cropped = crop_to_original(instance_result, crop_params, orig_shape)
    img_cropped = crop_to_original(padded_img, crop_params, orig_shape)
    return img_cropped, instance_cropped
