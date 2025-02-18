import sys
import json
import orthanc
import os

# Load configuration
config = json.loads(orthanc.GetConfiguration()).get('Fracture', {})
venv = config.get('VirtualEnv')

if venv:
    sys.path.insert(0, venv)

# Define constants
STONE_VERSION = '2024-08-31-StoneWebViewer-DICOM-SR'
VIEWER_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'viewer')
os.makedirs(VIEWER_DIR, exist_ok=True)

# MIME types
MIME_TYPES = {
    '.css': 'text/css',
    '.gif': 'image/gif',
    '.html': 'text/html',
    '.jpeg': 'image/jpeg',
    '.js': 'text/javascript',
    '.png': 'image/png',
}

import os
import io
import zipfile
import highdicom
import pydicom
import modelYolo
import dicom_sr
import download
# Load Stone Web Viewer
stone_assets = zipfile.ZipFile(os.path.join(VIEWER_DIR, f'{STONE_VERSION}.zip'))

def serve_stone_web_viewer(output, uri, **request):
    if request['method'] != 'GET':
        output.SendMethodNotAllowed('GET')
        return
    
    if uri.startswith('/fracture-viewer/'):
        path = f'{STONE_VERSION}/{uri[len("/fracture-viewer/"):]}'
    elif uri.startswith('/lcancer-viewer/'):
        path = f'{STONE_VERSION}/{uri[len("/lcancer-viewer/"):]}'
    else:
        output.SendHttpStatusCode(404)
        return
    
    extension = os.path.splitext(path)[1]
    mime = MIME_TYPES.get(extension, 'application/octet-stream')
    
    try:
        with stone_assets.open(path) as f:
            output.AnswerBuffer(f.read(), mime)
    except:
        output.SendHttpStatusCode(500)

orthanc.RegisterRestCallback('/fracture-viewer/(.*)', serve_stone_web_viewer)
orthanc.RegisterRestCallback('/lcancer-viewer/(.*)', serve_stone_web_viewer)

# Load models
orthanc.LogWarning('Loading the Yolo models for detection')
fracture_model = modelYolo.load_yolo_model("models/fracture.pt")
lcancer_model = modelYolo.load_yolo_model("models/lcancer.pt")

# Extend Orthanc Explorer

with open(os.path.join(os.path.dirname(__file__), 'OrthancExplorer.js'), 'r') as f:
    orthanc.ExtendOrthancExplorer(f.read())

def execute_inference(output, uri, **request):
    if request['method'] != 'POST':
        output.SendMethodNotAllowed('POST')
        return
    
    body = json.loads(request['body'])
    f = orthanc.GetDicomForInstance(body['instance'])
    dicom = pydicom.dcmread(io.BytesIO(f))
    
    if len(dicom.pixel_array.shape) != 2:
        orthanc.LogError(f'Not a graylevel instance: {body["instance"]}')
        output.SendHttpStatusCode(400)
        return
    
    if uri == '/fracture-apply':
        result = dicom_sr.applyYolo(fracture_model, dicom, minimum_score=0.2)
    elif uri == '/lcancer-apply':
        result = dicom_sr.applyYolo(lcancer_model, dicom, minimum_score=0.2)
    else:
        output.SendHttpStatusCode(404)
        return
    
    with io.BytesIO() as f:
        pydicom.dcmwrite(f, result)
        f.seek(0)
        content = f.read()
    
    output.AnswerBuffer(orthanc.RestApiPost('/instances', content), 'application/json')

orthanc.RegisterRestCallback('/fracture-apply', execute_inference)
orthanc.RegisterRestCallback('/lcancer-apply', execute_inference)
