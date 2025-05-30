import torch
import torch.nn as nn
from torchvision import models, transforms

class SoilTextureModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = []
        
        # Load full ResNet50 architecture as saved in the checkpoint
        self.resnet = models.resnet50(pretrained=False)
        
        # Modify the final layer to match our number of classes
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

def load_soil_model(model_path='soil_model_state_dict_v4.pth'):
    try:
        # Load with weights_only=False for compatibility
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Initialize model with correct number of classes
        model = SoilTextureModel(num_classes=checkpoint['num_classes'])
        
        # Handle both full checkpoints and state_dict only
        if 'model_state_dict' in checkpoint:
            # Load the complete state dict
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Direct state_dict load
            model.load_state_dict(checkpoint)
            
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