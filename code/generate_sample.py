import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from torchvision import transforms

#1. Set target car here
TARGET_CAR_NAME = 'Audi S5 Coupe' 

#2. Load the training data
df = pd.read_csv('train_split.csv')

#3. Dynamically filter for the target car
target_subset = df[df['class_name'].str.contains(TARGET_CAR_NAME, case=False, na=False)]

#Failsafe in case of a typo in the target name
if target_subset.empty:
    print(f"Error: Could not find '{TARGET_CAR_NAME}' in the dataset.")
    exit()

#Grab the very first image of target car
row = target_subset.iloc[0] 
actual_class_name = row['class_name']

#4. Extract image
img_path = os.path.join('cars_train/cars_train', row['filename'])
image = Image.open(img_path).convert('RGB')

#5. Extract bounding box
x1, y1, x2, y2 = row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2']

#6. Create the cropped and resized version
cropped_image = image.crop((x1, y1, x2, y2)).resize((224, 224))

#7. Panel 3: Apply PyTorch Augmentations
#Force the flip to 100% here
aug_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1.0),
    transforms.ColorJitter(brightness=0.2, contrast=0.2)
])
augmented_image = aug_transforms(cropped_image)

#8. Plotting the 3 Panels
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

#Panel 1: Original with Bounding Box
ax1.imshow(image)
rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
ax1.add_patch(rect)
ax1.set_title(f"1. Original Image\nClass: {actual_class_name}")
ax1.axis('off')

#Panel 2: Cleaned Input
ax2.imshow(cropped_image)
ax2.set_title("2. Network Input\n(Cropped & 224x224)")
ax2.axis('off')

#Panel 3: Augmented Input
ax3.imshow(augmented_image)
ax3.set_title("3. Augmented Input\n(Flipped & Color Jittered)")
ax3.axis('off')

plt.tight_layout()
plt.savefig('cleaned_sample.png', dpi=300, bbox_inches='tight')
print(f"Successfully saved 3-panel cleaned_sample.png using: {actual_class_name}")