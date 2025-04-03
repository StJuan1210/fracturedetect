import os
import re
import numpy as np
import torch
from torchvision.transforms import functional as F
from ultralytics import YOLO
from sahi.predict import get_sliced_prediction
from sahi import AutoDetectionModel
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

class ResizeBetter:
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, image):
        _, h, w = image.shape  # C, H, W
        scale = self.min_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        image = F.resize(image, (new_h, new_w), interpolation=F.InterpolationMode.BILINEAR)

        pad_w = (self.min_size - new_w) // 2
        pad_h = (self.min_size - new_h) // 2

        image = F.pad(image, (pad_w, pad_h, self.min_size - new_w - pad_w, self.min_size - new_h - pad_h))

        return image

def load_yolo_model(weights_path):
    model = YOLO(weights_path)
    model.eval()
    print(f"YOLO model loaded from {weights_path}")
    return model

def load_sahi_model(weights_path):
    detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov11', 
    model_path=weights_path,
    confidence_threshold=0.25,
    device='0' #or cpu
        )
    return detection_model

def dicom_to_tensor(dicom):
    assert len(dicom.pixel_array.shape) == 2
    im_array = np.stack((dicom.pixel_array,) * 3, axis=-1)
    image_tensor = torch.tensor(im_array.astype(np.float32).transpose(2, 0, 1))
    assert len(image_tensor.shape) == 3
    return image_tensor

def dicom_to_numpy(dicom):
    assert len(dicom.pixel_array.shape) == 2
    im_array = np.stack((dicom.pixel_array,) * 3, axis=-1)
    return im_array

def apply_model_to_dicom(model, dicom, rescale_boxes=True):
    image_tensor = dicom_to_tensor(dicom)
    image_tensor = ResizeBetter(640)(image_tensor)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Normalize tensor
    image_tensor = image_tensor / image_tensor.max()

    # Run inference with YOLO
    results = model.predict(image_tensor, device='cpu')

    detections = results[0].boxes.xyxy.cpu().numpy()
    confidences = results[0].boxes.conf.cpu().numpy()
    labels = results[0].boxes.cls.cpu().numpy()

    if rescale_boxes:
        original_width = dicom.pixel_array.shape[1]
        original_height = dicom.pixel_array.shape[0]
        padded_width = image_tensor.shape[3]
        padded_height = image_tensor.shape[2]

        scale = min(640 / original_width, 640 / original_height)
        resized_width = int(original_width * scale)
        resized_height = int(original_height * scale)

        pad_w = (padded_width - resized_width) // 2
        pad_h = (padded_height - resized_height) // 2

        detections[:, [0, 2]] -= pad_w
        detections[:, [1, 3]] -= pad_h

        scale_x = original_width / resized_width
        scale_y = original_height / resized_height

        detections[:, [0, 2]] *= scale_x
        detections[:, [1, 3]] *= scale_y

    return {
        'boxes': detections,
        'confidences': confidences,
        'labels': labels
    }

def apply_model_to_dicom_sahi(detection_model, dicom):
    image = dicom_to_numpy(dicom)
    result = get_sliced_prediction(
        image,
        detection_model,
        slice_height=256,
        slice_width=256,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
        verbose=False
    ).object_prediction_list
    boxes = []
    confidences = []
    labels = []
    for obj in result:
        boxes.append(obj.bbox.to_xyxy())
        confidences.append(obj.score.value)
        labels.append(obj.category.id)

    detected = {
        'boxes': np.array(boxes),
        'confidences': np.array(confidences),
        'labels': np.array(labels)
    }
    return detected