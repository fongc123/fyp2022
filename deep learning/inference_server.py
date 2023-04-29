# Flask server for model inference
from orange_peels import Constants as c
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from flask import Flask, request, jsonify
from torchvision import transforms
from torchvision.io import read_image
from torchvision.models import resnet50
from torch.utils.data import DataLoader, Dataset

MODEL_PATH = "./resnet50_full_no_kfold.pt"

transform = transforms.Compose([
    transforms.Resize((int(c.IMG_SIZE[0]*c.IMG_MAG), int(c.IMG_SIZE[1]*c.IMG_MAG))),
    transforms.Normalize(mean=c.RESNET_MEAN, std=c.RESNET_STD)
])

model = torch.load(MODEL_PATH).to(torch.device("cpu"))
model.eval()

app = Flask(__name__)

@app.route('/', methods = [ 'GET' ])
def hello():
    return "hello world"

@app.route('/api/predict', methods = [ 'POST' ])
def predict():
    image = request.files["image"]
    filename = "image.jpg"
    image.save(filename)
    image = transform(read_image(filename).float()).unsqueeze(0)
    os.remove(filename)

    with torch.no_grad():
        output = model(image)
        _, pred = torch.max(output, 1)
        pred = pred.item()

    return jsonify({ "prediction" : pred })

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)