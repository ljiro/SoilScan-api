import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.ensemble import RandomForestClassifier
import joblib

class SoilTextureModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
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
        self.device = torch.device("cpu")
        self.rf_classifier = None
        self.class_names = []

    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        rf_feats = self.rf_features(features)
        out = self.classifier(features)
        return out, None, rf_feats

def load_soil_model(model_path='soil_model_state_dict_v4.pth', rf_path='random_forest_v4.pkl'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        model = SoilTextureModel(num_classes=checkpoint['num_classes'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.class_names = [str(c).strip() for c in checkpoint['class_names']]  # Clean whitespace
        
        # Load RF classifier
        model.rf_classifier = joblib.load(rf_path)
        
        # Create mapping between RF numeric classes and model class names
        if hasattr(model.rf_classifier, 'classes_'):
            rf_classes = [str(c).strip() for c in model.rf_classifier.classes_]
            
            # If RF uses numeric classes (0-9) and we have text classes
            if all(c.isdigit() for c in rf_classes) and not any(c.isdigit() for c in model.class_names):
                # Create 1:1 mapping assuming same number of classes
                if len(rf_classes) == len(model.class_names):
                    model.class_mapping = {int(num): name for num, name in zip(rf_classes, model.class_names)}
                    model.rf_to_model_mapping = {i: i for i in range(len(rf_classes))}
                else:
                    raise ValueError("Class count mismatch between numeric RF and text model classes")
        
        model.eval()
        return model
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")
        
# Image transformation
transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])