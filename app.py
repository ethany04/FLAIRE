from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image
import torch
import io
from model import CNNModel
from werkzeug.utils import secure_filename
import time
import os

app = Flask(__name__)


label_mapping = {0: 'Accident', 1: 'Non Accident'}

# Load your model
model = torch.load('0.95.pth')
model.eval()

# Define the transform
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'frames' not in request.files:
        return 'No file part'
    
    files = request.files.getlist('frames')

    results = []
    for file in files:
        if file.filename == '':
            return 'No selected file'
        filename = secure_filename(file.filename)
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = transform(image)
        image = image.unsqueeze(0)  

        start_time = time.time()

        #prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            prediction_label = label_mapping[predicted.item()]  
        
        end_time = time.time()
        total_time = end_time - start_time
        fps = 1.0 / total_time if total_time > 0 else "NaN"

        results.append((filename, prediction_label, total_time, fps))


    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
