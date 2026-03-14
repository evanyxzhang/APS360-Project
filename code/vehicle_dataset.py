import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

#Custom Dataset Class
class VehicleDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        #1 Get the row from the CSV
        row = self.data_frame.iloc[idx]
        
        #2 Load the image
        img_path = os.path.join(self.img_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')

        #3 Apply Bounding Box prior to resizing/augmentation
        bbox = (row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2'])
        image = image.crop(bbox)

        #4 Apply PyTorch Transforms (if applicable)
        if self.transform:
            image = self.transform(image)

        #5 Adjust to 0-indexed classes
        label = row['class_id'] - 1 
        
        return image, label
    
#Transformation Pipeline

#Training Pipeline (heavy augmentations)
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),                 # 50% chance to flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2),   # Simulate different lighting
    transforms.ToTensor(),                                  # Convert to PyTorch FloatTensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],        # Standard ImageNet normalization
                         std=[0.229, 0.224, 0.225])
])

#Validation Pipeline (no randomness)
val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

#Test
if __name__ == "__main__":
    print("Initializing Datasets...")
    
    # Instantiate the Datasets
    train_dataset = VehicleDataset(csv_file='train_split.csv', img_dir='cars_train/cars_train', transform=train_transforms)
    val_dataset = VehicleDataset(csv_file='val_split.csv', img_dir='cars_train/cars_train', transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    images, labels = next(iter(train_loader))
    
    print("--- Pipeline Test Successful ---")
    print(f"Batch Image Tensor Shape: {images.shape}")
    print(f"Batch Label Tensor Shape: {labels.shape}")