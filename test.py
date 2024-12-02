import modelYolo
import cv2
import numpy as np
import pydicom
import dicom_sr
import io


print("Hello World")
model = modelYolo.load_yolo_model("models/best.pt")
# model = model.load_retina_net()
f = io.BytesIO(open("sample-images/sample.dcm", "rb").read())
dicom = pydicom.dcmread(io.BytesIO(f.read()))
result = dicom_sr.applyYolo(model, dicom, minimum_score=0.2)
print(result)
cv2.imwrite("Image.png", dicom.pixel_array)
# with io.BytesIO() as f:   
#     pydicom.dcmwrite(f, result)
#     f.seek(0)
#     content = np.asarray(f.read(), dtype=np.uint8)
#     # Overlay the result on the dicom file
#     img = cv2.imdecode(content, cv2.IMREAD_COLOR)
#     cv2.imwrite("Image.png", img)


