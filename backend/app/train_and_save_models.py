import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, swin_t
from tqdm import tqdm

# =========================
# CONFIGURATION
# =========================
DATA_DIR = "dataset"  # folder with 'train/' and 'val/' subfolders
SAVE_DIR = "models"
os.makedirs(SAVE_DIR, exist_ok=True)
BATCH_SIZE = 16
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# DATA
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
num_classes = len(train_ds.classes)
print("Classes:", train_ds.classes)

# =========================
# DEFINE MODELS
# =========================
def get_vit(num_classes):
    model = vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model

def get_swin(num_classes):
    model = swin_t(weights=None)
    model.head = nn.Linear(model.head.in_features, num_classes)
    return model

vit_model = get_vit(num_classes).to(DEVICE)
swin_model = get_swin(num_classes).to(DEVICE)

criterion = nn.CrossEntropyLoss()
vit_opt = torch.optim.Adam(vit_model.parameters(), lr=LR)
swin_opt = torch.optim.Adam(swin_model.parameters(), lr=LR)

# =========================
# TRAINING LOOP WITH PROGRESS AND EARLY STOPPING ON VAL ACC
# =========================
def train_model(model, optimizer, name, patience=3):
    best_val_acc = 0
    counter = 0  # Early stopping counter

    for epoch in range(EPOCHS):
        # ---- Training ----
        model.train()
        train_loss_sum, train_correct, train_total = 0, 0, 0
        loop = tqdm(train_loader, desc=f"{name} Epoch {epoch+1}/{EPOCHS}", unit="batch")
        for x, y in loop:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            train_correct += (out.argmax(1) == y).sum().item()
            train_total += y.size(0)

            loop.set_postfix(loss=loss.item(), acc=train_correct/train_total)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total

        # ---- Validation ----
        model.eval()
        val_loss_sum, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                loss = criterion(out, y)
                val_loss_sum += loss.item() * x.size(0)
                val_correct += (out.argmax(1) == y).sum().item()
                val_total += y.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # ---- Early stopping & saving best model based on val_acc ----
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"{name}_best.pth"))
            print(f"✅ Saved {name}_best.pth (Val Acc improved)")
        else:
            counter += 1
            print(f"⚠️  Validation accuracy did not improve (counter {counter}/{patience})")
            if counter >= patience:
                print("Early stopping triggered!")
                break

# =========================
# TRAIN MODELS
# =========================
train_model(vit_model, vit_opt, "vit")
train_model(swin_model, swin_opt, "swin")
