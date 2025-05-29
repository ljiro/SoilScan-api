import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Define the same model architecture
class SoilTextureModel(nn.Module):
    def __init__(self, num_classes):
        super(SoilTextureModel, self).__init__()
        self.num_classes = num_classes
        self.resnet = models.resnet50(pretrained=False)
        
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.fc = nn.Sequential(
            nn.Linear(2048, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.rf_features = nn.Linear(128, 64)
        self.classifier = nn.Linear(128, num_classes)
        self.rf_classifier = RandomForestClassifier(n_estimators=50)

    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        rf_feats = self.rf_features(features)
        out = self.classifier(features)
        return out, None, rf_feats

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load saved checkpoint
    checkpoint = torch.load('soil_model_state_dict.pth', map_location=device)
    
    # Initialize model
    model = SoilTextureModel(num_classes=checkpoint['num_classes'])
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.rf_classifier = checkpoint['rf_classifier']
    model.class_names = checkpoint['class_names']
    
    model.eval()
    return model

# Preprocessing transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Example usage
model = load_model()
print("Model loaded successfully with classes:", model.class_names)

# For prediction
def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output, _, _ = model(image)
        _, predicted = torch.max(output, 1)
    return model.class_names[predicted.item()]

# Test prediction
print("Predicted class:", predict('test_image.jpg'))