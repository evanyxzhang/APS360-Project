import torch
import torch.nn as nn
from torchvision import models
from torch.utils.data import DataLoader
from vehicle_dataset import VehicleDataset, val_transforms
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Hunting for reasonable mistakes (Same Manufacturer, Wrong Model/Year)...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
val_dataset = VehicleDataset(csv_file='val_split.csv', img_dir='cars_train/cars_train', transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

df_val = pd.read_csv('val_split.csv')
class_mapping = {row['class_id'] - 1: row['class_name'] for _, row in df_val.iterrows()}

model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 196)
model.load_state_dict(torch.load('vmmr_resnet50_model.pth', map_location=device))
model.to(device)
model.eval()

mistakes_to_find = 3
found_images = []
found_titles = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        
        misclassified_indices = (preds != labels).nonzero(as_tuple=True)[0]
        
        for idx in misclassified_indices:
            true_label = class_mapping[labels[idx].item()]
            predicted_label = class_mapping[preds[idx].item()]
            
            # The filter: Only keep the mistake if the first word (Make) matches
            true_make = true_label.split()[0]
            pred_make = predicted_label.split()[0]
            
            if true_make == pred_make: 
                img = images[idx].cpu().numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)
                
                found_images.append(img)
                found_titles.append(f"Actual: {true_label}\nPred: {predicted_label}")
                
                if len(found_images) >= mistakes_to_find:
                    break
        if len(found_images) >= mistakes_to_find:
            break

# Plot the 3 mistakes side-by-side
if found_images:
    fig, axes = plt.subplots(1, len(found_images), figsize=(15, 5))
    if len(found_images) == 1:
        axes = [axes]
    for ax, img, title in zip(axes, found_images, found_titles):
        ax.imshow(img)
        ax.set_title(title, color='red', fontsize=10, fontweight='bold')
        ax.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("Couldn't find any same-brand mistakes in this pass!")