import torch
import torch.nn as nn
from torchvision import models, transforms

class SoilTextureModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = []
        
        # Use ResNet50 architecture to match training
        self.resnet = models.resnet50(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove final FC layer
        
        # Custom classifier head to match training
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        
    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        return self.classifier(features)

def load_soil_model(model_path='soil_model_state_dict_v5.pth'):
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Initialize model with correct number of classes
        model = SoilTextureModel(num_classes=checkpoint['num_classes'])
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        # Load class names
        model.class_names = [str(c).strip() for c in checkpoint['class_names']]
        
        model.eval()
        model.to(model.device)
        return model
        
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

# Image transformation (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])