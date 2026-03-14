import scipy.io
import pandas as pd
import os

print("Loading MATLAB files...")

#1. Load the Meta Data (Class Names)
meta_mat = scipy.io.loadmat('car_devkit/devkit/cars_meta.mat')

# Flatten the nested MATLAB arrays into a clean Python list of strings
class_names = [c[0] for c in meta_mat['class_names'][0]]

#2. Load Training Annotations
train_mat = scipy.io.loadmat('car_devkit/devkit/cars_train_annos.mat')
annotations = train_mat['annotations'][0]

#3. Extract data
print(f"Extracting {len(annotations)} annotations...")
data = []
for anno in annotations:
    class_id = anno['class'][0][0]
    
    data.append({
        'filename': anno['fname'][0],           # e.g., '00001.jpg'
        'bbox_x1': anno['bbox_x1'][0][0],       # Min x-value
        'bbox_y1': anno['bbox_y1'][0][0],       # Min y-value
        'bbox_x2': anno['bbox_x2'][0][0],       # Max x-value
        'bbox_y2': anno['bbox_y2'][0][0],       # Max y-value
        'class_id': class_id,                   # Integer ID (1 to 196)
        'class_name': class_names[class_id - 1] # Human-readable name
    })

#4. Convert to DataFrame and save to CSV
df_train = pd.DataFrame(data)
csv_path = 'train_annotations.csv'
df_train.to_csv(csv_path, index=False)

print("\nSuccess! Here are the first 5 rows of your new dataset:")
print(df_train.head())
print(f"\nSaved full table to: {os.path.abspath(csv_path)}")