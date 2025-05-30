import torch
import torch.nn as nn
from torchvision import models, transforms

class SoilTextureModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = []
        
        # Original architecture that matches the saved model
        self.resnet = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Add the custom layers that were in the original model
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

    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        rf_feats = self.rf_features(features)
        out = self.classifier(features)
        return out, None, rf_feats

def load_soil_model(model_path='soil_model_state_dict_v4.pth'):
    try:
        # Allow loading with weights_only=False since we trust the source
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Initialize model with correct number of classes
        model = SoilTextureModel(num_classes=checkpoint['num_classes'])
        
        # Handle both full checkpoints and state_dict only
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # Load the state dict, ignoring unexpected/missing keys
        model.load_state_dict(state_dict, strict=False)
        
        # Load class names
        model.class_names = [str(c).strip() for c in checkpoint['class_names']]
        
        model.eval()
        model.to(model.device)
        return model
        
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Standard size for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])