from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


# Define the model architecture
class CervicalCancerCNN(nn.Module):
    def __init__(self, num_classes=4):
        super(CervicalCancerCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Load the pre-trained model
MODEL_PATH = os.path.join('model', 'best_model_20250529-200357.pth')
device = torch.device('cpu')
try:
    model = CervicalCancerCNN(num_classes=4).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Define class names (update based on your dataset)
CLASS_NAMES = ['HSIL', 'LSIL', 'NILM', 'SCC']  # Adjust as per your classes
# Define image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Function to preprocess the uploaded image
def preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = preprocess(img)
        img = img.unsqueeze(0)
        return img.to(device)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        raise


# Home route
@app.route('/')
def index():
    return render_template('index.html')


# Upload and predict route
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', message='No selected file')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            img_tensor = preprocess_image(file_path)
            with torch.no_grad():
                output = model(img_tensor)
                probabilities = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                predicted_class = probabilities.argmax()
                predicted_label = CLASS_NAMES[predicted_class]
                confidence = probabilities[predicted_class] * 100
            os.remove(file_path)
            return render_template('index.html',
                                   message=f'Classification: {predicted_label} (Confidence: {confidence:.2f}%)')
        except Exception as e:
            os.remove(file_path)
            return render_template('index.html', message=f'Error processing image: {str(e)}')
    return render_template('index.html', message='Invalid file type')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)