import os
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ViTForImageClassification, SwinForImageClassification
from torchvision import transforms

# -------------------------------
# Configuration
# -------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")

CLASS_NAMES = ["BacterialBlights", "Healthy", "Mosaic", "RedRot", "Rust", "Yellow"]

# -------------------------------
# Load ViT model
# -------------------------------
def load_vit():
    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "vit_best.pth"), map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# -------------------------------
# Load Swin model
# -------------------------------
def load_swin():
    model = SwinForImageClassification.from_pretrained(
        "microsoft/swin-base-patch4-window7-224",
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True
    )
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "swin_best.pth"), map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# -------------------------------
# Initialize models
# -------------------------------
vit_model = load_vit()
swin_model = load_swin()

# -------------------------------
# Image Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------------------------------
# Ensemble Prediction
# -------------------------------
def predict_image(image: Image.Image):
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        vit_probs = torch.softmax(vit_model(image).logits, dim=1)
        swin_probs = torch.softmax(swin_model(image).logits, dim=1)
        avg_probs = (vit_probs + swin_probs) / 2
        conf, idx = torch.max(avg_probs, 1)
    return CLASS_NAMES[idx.item()], float(conf.item())
