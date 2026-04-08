import os
import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
import pandas as pd
from vehicle_dataset import val_transforms

print("Loading model for real-world testing...")

# 1. Setup device and class mapping
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df_val = pd.read_csv('val_split.csv')
class_mapping = {row['class_id'] - 1: row['class_name'] for _, row in df_val.iterrows()}

# 2. Load your saved model
model = models.resnet50(weights=None)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 196)
model.load_state_dict(torch.load('vmmr_resnet50_model.pth', map_location=device))
model.to(device)
model.eval()

# 3. Target folder
test_dir = 'test_cars'

# 4. Run inference on custom images
if not os.path.exists(test_dir):
    print(f"Please create a folder named '{test_dir}' and add some images.")
else:
    print("\n--- Predictions ---")
    with torch.no_grad():
        for filename in os.listdir(test_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(test_dir, filename)
                
                # Load and transform image
                image = Image.open(img_path).convert('RGB')
                # Note: We resize directly here since we don't have bounding boxes for street view
                image = image.resize((224, 224)) 
                input_tensor = val_transforms(image).unsqueeze(0).to(device)
                
                # Predict
                outputs = model(input_tensor)
                _, preds = torch.max(outputs, 1)
                
                predicted_label = class_mapping[preds.item()]
                print(f"File: {filename} --> Predicted: {predicted_label}")