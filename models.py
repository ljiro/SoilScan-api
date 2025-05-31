import torch
import torch.nn as nn
from torchvision import models, transforms

class SoilTextureModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = []
        
        # Modified architecture with proper feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # Additional layers to reduce dimensions
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.AdaptiveAvgPool2d((1, 1))  # This will output [batch, 256, 1, 1]
        )
        
        # Modified classifier head - MAKE SURE THIS PARENTHESIS IS CLOSED
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),  # Matches the 256 features from AdaptiveAvgPool
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = features.view(features.size(0), -1)  # Flatten to [batch, 256]
        out = self.classifier(features)
        return out
    
def load_soil_model(model_path='soil_model_state_dict_v4.pth'):
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