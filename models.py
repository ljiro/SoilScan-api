import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.serialization import add_safe_globals
import sklearn.ensemble._forest

class SoilTextureModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = []
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=False)
        
        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Direct classification with ResNet50
        return self.resnet(x)

def load_soil_model(model_path='soil_model_state_dict_v4.pth'):
    try:
        # Allow loading RandomForestClassifier if it exists in the checkpoint
        add_safe_globals([sklearn.ensemble._forest.RandomForestClassifier])
        
        # Load with weights_only=False since we trust our source
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        model = SoilTextureModel(num_classes=checkpoint['num_classes'])
        
        # Handle both old and new model formats
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:  # If it's a direct model save
            model.load_state_dict(checkpoint)
            
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