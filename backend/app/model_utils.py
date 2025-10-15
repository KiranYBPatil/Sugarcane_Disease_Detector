import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import vit_b_16, swin_t
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

# Correct 6 classes
CLASS_NAMES = ["BacterialBlights", "Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]

def load_vit():
    model = vit_b_16(weights=None)
    # Update head to match number of classes
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, len(CLASS_NAMES))
    # Load the trained checkpoint
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "vit_best.pth"), map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def load_swin():
    model = swin_t(weights=None)
    # Update head to match number of classes
    model.head = torch.nn.Linear(model.head.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "swin_best.pth"), map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# Load models
vit_model = load_vit()
swin_model = load_swin()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        vit_out = F.softmax(vit_model(image), dim=1)
        swin_out = F.softmax(swin_model(image), dim=1)
        avg = (vit_out + swin_out) / 2
        conf, idx = torch.max(avg, 1)
    return CLASS_NAMES[idx.item()], float(conf.item())
