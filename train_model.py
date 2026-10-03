"""
Handwritten Alphabet Recognition — Model Training Pipeline
============================================================
Trains a CNN on the EMNIST Letters dataset (26 letter classes: A-Z)
using PyTorch.

Saves the trained model to model/handwritten_model.pth
and a label mapping to model/label_mapping.json.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BATCH_SIZE = 128
EPOCHS = 10
IMG_SIZE = 28
NUM_CLASSES = 26  # EMNIST letters (A-Z)
LEARNING_RATE = 1e-3

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
DATA_DIR = os.path.join(BASE_DIR, "data")

LABEL_CHARS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ──────────────────────────────────────────────
# CNN Model
# ──────────────────────────────────────────────
class HandwrittenCNN(nn.Module):
    """
    CNN architecture for handwritten alphabet recognition.
    Conv2D blocks with BatchNorm, ReLU, MaxPool, and Dropout.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
        )

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


# ──────────────────────────────────────────────
# Functions
# ──────────────────────────────────────────────
def build_label_mapping():
    """Create and save the label-to-character mapping."""
    mapping = {str(i): ch for i, ch in enumerate(LABEL_CHARS)}
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, "label_mapping.json")
    with open(path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"✅ Label mapping saved → {path}", flush=True)
    return mapping


def load_data():
    """Load EMNIST Letters with images oriented upright."""
    print("📥 Loading EMNIST Letters dataset …", flush=True)

    transform = transforms.Compose([
        transforms.Lambda(
            lambda image: image.transpose(Image.Transpose.TRANSPOSE)
        ),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.EMNIST(
        root=DATA_DIR, split="letters", train=True, download=True, transform=transform
    )
    test_dataset = datasets.EMNIST(
        root=DATA_DIR, split="letters", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print(f"   Training samples : {len(train_dataset):,}", flush=True)
    print(f"   Test samples     : {len(test_dataset):,}", flush=True)
    print(f"   Classes          : {NUM_CLASSES} (letters A-Z)", flush=True)
    print(f"   Batch size       : {BATCH_SIZE}", flush=True)

    return train_loader, test_loader


def train_one_epoch(model, loader, criterion, optimizer, epoch, total_epochs):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion):
    """Evaluate on test set."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def plot_history(history):
    """Save accuracy and loss curves."""
    os.makedirs(PLOTS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(history["train_acc"]) + 1)

    ax1.plot(epochs_range, history["train_acc"], label="Train", linewidth=2)
    ax1.plot(epochs_range, history["val_acc"], label="Validation", linewidth=2)
    ax1.set_title("Model Accuracy", fontsize=14, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs_range, history["train_loss"], label="Train", linewidth=2)
    ax2.plot(epochs_range, history["val_loss"], label="Validation", linewidth=2)
    ax2.set_title("Model Loss", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "training_history.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"📊 Training plots saved → {path}", flush=True)


def main():
    print("=" * 60, flush=True)
    print("🤖  Handwritten Alphabet Recognition — Training", flush=True)
    print(f"   Device: {DEVICE}", flush=True)
    print("=" * 60, flush=True)

    # 1. Label mapping
    build_label_mapping()

    # 2. Data
    train_loader, test_loader = load_data()

    # 3. Model
    model = HandwrittenCNN(NUM_CLASSES).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model parameters: {total_params:,}", flush=True)
    print(model, flush=True)

    # 4. Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
    )

    # 5. Training loop
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, "handwritten_model.pth")

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    patience_counter = 0
    patience = 5

    print("\n🚀 Starting training …\n", flush=True)

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, epoch, EPOCHS
        )
        val_loss, val_acc = evaluate(model, test_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Epoch {epoch:2d}/{EPOCHS} │ "
            f"Train Loss: {train_loss:.4f}  Acc: {train_acc*100:.2f}% │ "
            f"Val Loss: {val_loss:.4f}  Acc: {val_acc*100:.2f}% │ "
            f"LR: {lr:.6f}",
            flush=True,
        )

        scheduler.step(val_loss)

        # Early stopping + best model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "val_acc": val_acc,
                "epoch": epoch,
            }, model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch} (patience={patience})", flush=True)
                break

    # 6. Final results
    print(f"\n✅ Best Validation Accuracy: {best_val_acc * 100:.2f}%", flush=True)
    print(f"💾 Model saved → {model_path}", flush=True)

    # 7. Plots
    plot_history(history)

    print("\n" + "=" * 60, flush=True)
    print("🎉  Training complete!", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
