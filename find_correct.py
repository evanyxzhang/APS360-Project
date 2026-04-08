import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from vehicle_dataset import VehicleDataset, val_transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_dataset = VehicleDataset(csv_file='val_split.csv', img_dir='cars_train/cars_train', transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

df_val = pd.read_csv('val_split.csv')
class_mapping = {row['class_id'] - 1: row['class_name'] for _, row in df_val.iterrows()}

model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 196)
model.load_state_dict(torch.load('vmmr_resnet50_model.pth', map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        # Find where prediction MATCHES the actual label
        correct_indices = (preds == labels).nonzero(as_tuple=True)[0]
        
        if len(correct_indices) > 0:
            idx = correct_indices[0]
            img = images[idx].cpu().numpy().transpose((1, 2, 0))
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = np.clip(std * img + mean, 0, 1)
            
            true_label = class_mapping[labels[idx].item()]
            
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.title(f"Actual & Predicted:\n{true_label}", color='green', fontweight='bold')
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            break