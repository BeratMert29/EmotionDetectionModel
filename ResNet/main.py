import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

# Define emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

IMG_SIZE = 48
BATCH_SIZE = 64
EPOCHS = 40
NUM_CLASSES = 7

# Data augmentation
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.1)),
    transforms.ToTensor()
])

val_test_transforms = transforms.Compose([transforms.ToTensor()])


def load_data():
    X_train = np.load("train_X.npy")
    y_train = np.load("train_y.npy")
    X_test = np.load("test_X.npy")
    y_test = np.load("test_y.npy")

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    X_train = torch.tensor(X_train).float().unsqueeze(1)
    X_val = torch.tensor(X_val).float().unsqueeze(1)
    X_test = torch.tensor(X_test).float().unsqueeze(1)
    y_train = torch.tensor(y_train).long()
    y_val = torch.tensor(y_val).long()
    y_test = torch.tensor(y_test).long()

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, val_loader, test_loader

class EmotionModel(nn.Module):
    def __init__(self):
        super(EmotionModel, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),
            nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),
            nn.MaxPool2d(2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),
            nn.MaxPool2d(2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 3 * 3, 512),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.5),
            nn.Linear(256, 7)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.classifier(x)
        return x



def train_model(model, train_loader, val_loader, device):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = optim.Adam(model.parameters(), lr=0.0007, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.3)
    
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []
    
    # For confusion matrix
    all_preds = []
    all_labels = []
    
    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100 * train_correct / total
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # Validation
        model.eval()
        val_loss, val_correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                # Store predictions and labels for confusion matrix
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_acc = 100 * val_correct / total
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("------------------------------------------------------------------------------------------------")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        scheduler.step(val_loss)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot Loss
    plt.subplot(2, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, label='Validation Loss', linewidth=2)
    plt.title('Loss over Epochs', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Loss', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, EPOCHS)
    
    # Plot Accuracy
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', linewidth=2)
    plt.title('Accuracy over Epochs', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(1, EPOCHS)
    
    # Plot Learning Rate
    plt.subplot(2, 2, 3)
    plt.plot(epochs, learning_rates, linewidth=2, color='purple')
    plt.title('Learning Rate Schedule', fontsize=12, pad=10)
    plt.xlabel('Epoch', fontsize=10)
    plt.ylabel('Learning Rate', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.xlim(1, EPOCHS)
    
    # Confusion Matrix
    plt.subplot(2, 2, 4)
    cm = confusion_matrix(all_labels, all_preds)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalize
    plt.imshow(cm, cmap='Blues')
    plt.colorbar()
    plt.title('Normalized Confusion Matrix', fontsize=12, pad=10)
    plt.xlabel('Predicted', fontsize=10)
    plt.ylabel('True', fontsize=10)
    plt.xticks(range(len(emotions)), emotions, rotation=45)
    plt.yticks(range(len(emotions)), emotions)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create class-wise accuracy plot
    class_accuracies = np.diag(cm) * 100
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(emotions)), class_accuracies)
    plt.title('Class-wise Accuracy', fontsize=12, pad=10)
    plt.xlabel('Emotion', fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.xticks(range(len(emotions)), emotions, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('class_accuracy.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nTraining progress plots saved as:")
    print("- 'training_progress.png' (main training metrics)")
    print("- 'class_accuracy.png' (class-wise accuracy distribution)")

# Run training
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️ Using device: {device}")

    train_loader, val_loader, test_loader = load_data()
    model = EmotionModel().to(device)

    train_model(model, train_loader, val_loader, device)

    torch.save(model.state_dict(), "emotion_model.pth")
    print("✅ Model saved as emotion_model.pth")