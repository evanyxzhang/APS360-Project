import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from vehicle_dataset import VehicleDataset, train_transforms, val_transforms
from torch.utils.data import DataLoader

print("Setting up the ResNet-50 Neural Network...")

#1. Configuration
#Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

#2. Data Loading
train_dataset = VehicleDataset(csv_file='train_split.csv', img_dir='cars_train/cars_train', transform=train_transforms)
val_dataset = VehicleDataset(csv_file='val_split.csv', img_dir='cars_train/cars_train', transform=val_transforms)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

#3. Model Architecture (ResNet-50 Transfer Learning)
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 196)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#4. Training Loop
num_epochs=5
print("\nStarting Training...")

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i+1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_accuracy = 100 * correct / total
    print(f"---> Epoch {epoch+1} Validation Accuracy: {val_accuracy:.2f}% <---")

print("\nFinished Training!")
torch.save(model.state_dict(), 'vmmr_resnet50_model.pth')
print("Model saved to vmmr_resnet50_model.pth")