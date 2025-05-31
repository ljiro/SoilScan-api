import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.ensemble import RandomForestClassifier
import joblib

class SoilTextureModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])  # Remove original fc layer
        
        # Custom layers
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
        self.rf_classifier = RandomForestClassifier(n_estimators=100)
        self.class_names = []
        self.class_to_idx = {}

    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        rf_feats = self.rf_features(features)
        out = self.classifier(features)
        return out, None, rf_feats

def load_soil_model(model_path='soil_model_state_dict_v5.pth'):
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Initialize model
        model = SoilTextureModel(num_classes=checkpoint['num_classes'])
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load additional components
        model.class_names = checkpoint['class_names']
        model.class_to_idx = checkpoint['class_to_idx']
        model.rf_classifier = checkpoint['rf_classifier']
        
        # Set to evaluation mode
        model.eval()
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

# Image transformation (must match training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])