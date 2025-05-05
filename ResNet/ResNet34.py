import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

# Dataset Preparation
class PrepareDataset:
    def __init__(self):
        self.train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.RandomAffine(10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.val_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        self.train_dataset = datasets.ImageFolder(root="archive/train", transform=self.train_transform)
        self.val_dataset = datasets.ImageFolder(root="archive/test", transform=self.val_transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=64, shuffle=False)


# EfficientNet-B0 Model
class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=6):
        super(EfficientNetB0, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b0')

        # EfficientNet expects 3 channels; replicate grayscale to 3-channel input
        # You could alternatively rewrite the first conv layer, but this is simpler
        self.num_classes = num_classes
        self.model._fc = nn.Linear(self.model._fc.in_features, num_classes)

    def forward(self, x):
        # Expand grayscale 1-channel to 3-channel (repeat on channel dim)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.model(x)

    def TrainModel(self, train_loader, val_loader, device):
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.Adam(self.parameters(), lr=0.0007, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.3)

        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        self.to(device)

        for epoch in range(50):
            self.train()
            train_loss, train_correct, total = 0, 0, 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, preds = outputs.max(1)
                train_correct += preds.eq(labels).sum().item()
                total += labels.size(0)

            train_losses.append(train_loss / len(train_loader))
            train_accuracies.append(train_correct / total)

            self.eval()
            val_loss, val_correct, val_total = 0, 0, 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = self(images)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    _, preds = outputs.max(1)
                    val_correct += preds.eq(labels).sum().item()
                    val_total += labels.size(0)

            val_losses.append(val_loss / len(val_loader))
            val_accuracies.append(val_correct / val_total)

            scheduler.step(val_losses[-1])

            if val_losses[-1] < best_val_loss:
                best_val_loss = val_losses[-1]
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            print(f"Epoch {epoch + 1}/50, Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.4f}, "
                  f"Val Loss: {val_losses[-1]:.4f}, Val Acc: {val_accuracies[-1]:.4f}")

        return train_losses, train_accuracies, val_losses, val_accuracies


# Main
if __name__ == "__main__":
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        print(f"Number of CUDA devices: {torch.cuda.device_count()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = PrepareDataset()
    train_loader = dataset.train_loader
    val_loader = dataset.val_loader

    model = EfficientNetB0(num_classes=6).to(device)
    print(f"Model is on device: {next(model.parameters()).device}")
    print("Model initialized")

    print("Starting training...")
    train_losses, train_accuracies, val_losses, val_accuracies = model.TrainModel(
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
