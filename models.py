import torch
import torch.nn as nn
from torchvision import models, transforms

class SoilTextureModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.device = torch.device("cpu")
        self.class_names = []
        
        # Load pre-trained ResNet50
        self.resnet = models.resnet50(pretrained=False)
        
        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # Direct classification with ResNet50
        return self.resnet(x), None, None  # Maintaining same return signature for compatibility

def load_soil_model(model_path='soil_model_state_dict_v4.pth'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = SoilTextureModel(num_classes=checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.class_names = [str(c).strip() for c in checkpoint['class_names']]
        model.eval()
        model.to(device)
        return model
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")
        
# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Standard size for ResNet
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])