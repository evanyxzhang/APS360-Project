import os
import pandas as pd
import numpy as np
from PIL import Image
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import warnings

# Suppress convergence warnings from the SVM
warnings.filterwarnings('ignore')

print("Starting HOG + SVM Baseline Pipeline...")

#1. Feature Extraction Function
def extract_hog_features(csv_file, img_dir):
    df = pd.read_csv(csv_file)
    features = []
    labels = []
    
    print(f"Processing {len(df)} images from {csv_file}...")
    
    for idx, row in df.iterrows():
        img_path = os.path.join(img_dir, row['filename'])
        try:
            image = Image.open(img_path).convert('L')
        except FileNotFoundError:
            continue

        #Crop using the bounding box
        bbox = (row['bbox_x1'], row['bbox_y1'], row['bbox_x2'], row['bbox_y2'])
        image = image.crop(bbox)
        image = image.resize((128, 64))

        img_array = np.array(image)

        #Calculate HOG features
        fd = hog(img_array, orientations=9, pixels_per_cell=(8, 8),
                 cells_per_block=(2, 2), visualize=False)
        
        features.append(fd)
        labels.append(row['class_id'])

        if (idx + 1) % 1000 == 0:
            print(f"  ...extracted features for {idx + 1} images")

    return np.array(features), np.array(labels)

#2 Datasets
img_folder = 'cars_train/cars_train'

print("\n--- 1: Training Data ---")
X_train, y_train = extract_hog_features('train_split.csv', img_folder)
print("\n--- 2: Validation Data ---")
X_val, y_val = extract_hog_features('val_split.csv', img_folder)

#3 Train SVM Classifier
print("\n--- 3: Training the SVM ---")
print(f"Training on {X_train.shape[0]} samples with {X_train.shape[1]} HOG features each...")

svm_model = LinearSVC(C=1.0, random_state=24, max_iter=1000)
svm_model.fit(X_train, y_train)

#4 Test
print("\n--- 4: Testing ---")
# Predict on the training data
train_preds = svm_model.predict(X_train)
train_acc = accuracy_score(y_train, train_preds) * 100

# Predict on the unseen validation data
val_preds = svm_model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds) * 100

print(f"Baseline Training Accuracy:   {train_acc:.2f}%")
print(f"Baseline Validation Accuracy: {val_acc:.2f}%")
print("\nBaseline test complete.")