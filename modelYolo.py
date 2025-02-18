import download
import os
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import functional as F
from torchvision.transforms import Resize, Normalize
from PIL import Image
from ultralytics import YOLO 
SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')

os.makedirs(MODELS_DIR, exist_ok=True)

# download.get(os.path.join(MODELS_DIR, 'yolov11_weights.pth'),
#              'link',
#              size='size',
#              checksum='md5?')


# class ResizeBetter:
#     def __init__(self, min_size):
#         self.min_size = min_size

#     def __call__(self, image):
#         return Resize((self.min_size, self.min_size), 
#                         interpolation=F.InterpolationMode.BILINEAR)(image)

class ResizeBetter:
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, image):
        # Get original size
        _, h, w = image.shape # C, H, W
        # Determine scale factor
        scale = self.min_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize while maintaining aspect ratio
        image = F.resize(image, (new_h, new_w), interpolation=F.InterpolationMode.BILINEAR)

        # Compute padding
        pad_w = (self.min_size - new_w) // 2
        pad_h = (self.min_size - new_h) // 2

        # Apply padding (left, top, right, bottom)
        image = F.pad(image, 
                      (pad_w, 
                       pad_h, 
                       self.min_size - new_w - pad_w, 
                       self.min_size - new_h - pad_h))

        return image


def load_yolo_model(weights_path):
    # Load YOLOv11 model using the ultralytics library
    model = YOLO(weights_path)
    model.eval()  # Ensure the model is in evaluation mode
    print(f"YOLOv11 model loaded from {weights_path}")
    return model


def dicom_to_tensor(dicom, min_size):
    assert len(dicom.pixel_array.shape) == 2

    # Normalize pixel values to 0-255
    im_array = np.stack((dicom.pixel_array,) * 3, axis=-1)
    image_tensor = torch.tensor(im_array.astype(np.float32).transpose(2, 0, 1))

    # Resize and normalize
    image_tensor = ResizeBetter(min_size)(image_tensor)
    assert len(image_tensor.shape) == 3
    # torchvision.utils.save_image(image_tensor, 'test1.png')

    return image_tensor


def apply_model_to_dicom(model, dicom, rescale_boxes=True):
    image_tensor = dicom_to_tensor(dicom, 640)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Run inference with YOLOv11
    image_tensor*= (1.0/image_tensor.max())
    # torchvision.utils.save_image(image_tensor, 'test2.png')
    results = model.predict(image_tensor,device='cpu')

    # Process the output
    detections = results[0].boxes.xyxy.cpu().numpy()  # Get bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # Get confidence scores
    labels = results[0].boxes.cls.cpu().numpy()  # Get class labels

    # if rescale_boxes:
    #     original_width = dicom.pixel_array.shape[1]
    #     original_height = dicom.pixel_array.shape[0]
    #     resized_width = image_tensor.shape[3]
    #     resized_height = image_tensor.shape[2]

    #     scale_x = original_width / resized_width
    #     scale_y = original_height / resized_height

    #     # Scale boxes back to the original image size
    #     detections[:, [0, 2]] *= scale_x
    #     detections[:, [1, 3]] *= scale_y
    if rescale_boxes:
        original_width = dicom.pixel_array.shape[1]
        original_height = dicom.pixel_array.shape[0]
        padded_width = image_tensor.shape[3]  # Should be 640
        padded_height = image_tensor.shape[2]  # Should be 640

        # Calculate original resized image size before padding
        scale = min(640 / original_width, 640 / original_height)
        resized_width = int(original_width * scale)
        resized_height = int(original_height * scale)

        # Compute padding added
        pad_w = (padded_width - resized_width) // 2
        pad_h = (padded_height - resized_height) // 2

        # Adjust bounding boxes to remove padding
        detections[:, [0, 2]] -= pad_w
        detections[:, [1, 3]] -= pad_h

        # Scale boxes back to original image size
        scale_x = original_width / resized_width
        scale_y = original_height / resized_height

        detections[:, [0, 2]] *= scale_x
        detections[:, [1, 3]] *= scale_y
    return {
        'boxes': detections,
        'confidences': confidences,
        'labels': labels
    }
