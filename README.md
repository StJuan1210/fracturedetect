# Fracture and Lung Cancer detection Plugin for Orthanc


## Steps to add plugin

1. Clone the repo
2. Create a venv
    ```bash
    cd fracturedetect
    python3 -m venv .venv
    source ./.venv/bin/activate
    pip install -r requirements.txt
    ```
    If you use uv:
    ```bash
    uv venv --python 3.11.11
    uv sync
    ```
3. Edit the Orthanc config
   ```json
    "Plugins" : [
        ...
    ],
    "Python" : {
        "Path" : "./merge.py"
        // replace with fracture.py or lcancer.py if you want only one of them

    },
    "Fracture" : {
        "VirtualEnv" : "./venv/lib/python3.11/site-packages/" 
        //assumes python version is 3.11
    },
   ```
4. Create a folder called viewer and place the zipfile of stone web viewer binaries there. Plugin works with 2024-08-31 version of the viewer

### Notes
Models by default have to be in ```.pt``` format. To use ONNX or TensorRT for device specific speedups:

In ```./.venv/lib/python3.11/site-packages/sahi/models/ultralytics.py```

Change: 
```python
try:
    model = YOLO(self.model_path)
    model.to(self.device)
    self.set_model(model)
except Exception as e:
    raise TypeError("model_path is not a valid Ultralytics model path: ", e)
```
To:
```python
try:
    model = YOLO(self.model_path)
    if self.model_path.endswith(".pt"):
        model.to(self.device)
    self.set_model(model)
except Exception as e:
    raise TypeError("model_path is not a valid Ultralytics model path: ", e)
```