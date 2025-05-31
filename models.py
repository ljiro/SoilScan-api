import torch
import torch.nn as nn
from torchvision import models, transforms  # Fixed import here
from PIL import Image

# yes 
class SoilTextureModel(nn.Module):
    def __init__(self, num_classes):
        super(SoilTextureModel, self).__init__()
        self.num_classes = num_classes
        self.class_names = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load pre-trained ResNet50
        self.base_model = models.resnet50(pretrained=True)
        
        # Freeze early layers
        for param in list(self.base_model.parameters())[:-50]:
            param.requires_grad = False

        # Replace the final fully connected layer
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes))
        
        self.to(self.device)

    def forward(self, x):
        return self.base_model(x)

def load_soil_model(model_path='soil_texture_classifier.pth'):
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Initialize model
        model = SoilTextureModel(num_classes=checkpoint['num_classes'])
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load additional components
        model.class_names = checkpoint['class_names']
        
        # Set to evaluation mode
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