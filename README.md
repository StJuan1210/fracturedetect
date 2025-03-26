# Fracture and Lung Cancer detection Plugin for orthanc


## Steps to add plugin

1. Clone the repo
2. Create a venv
    ```bash
    cd fracturedetect
    python3 -m venv .venv
    source ./.venv/bin/activate
    pip install -r requirements.txt
    ```
3. Edit the Orthanc config
   ```json
    "Plugins" : [
        ...
    ],
    "Python" : {
        "Path" : "./merge.py"

    },
    "Fracture" : {
        "VirtualEnv" : "./venv/lib/python3.11/site-packages/" 
        //assumes python version is 3.11
    },
   ```

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