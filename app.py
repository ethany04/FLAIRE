from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image
import torch
import io
from model import CNNModel
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename

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
    if 'frame' not in request.files:
        return 'No file part'
    file = request.files['frame']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = transform(image)
        image = image.unsqueeze(0)  # Add batch dimension

        # Make a prediction
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            prediction_label = label_mapping[predicted.item()]  

    return render_template('result.html', label=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)
