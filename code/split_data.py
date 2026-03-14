import pandas as pd
from sklearn.model_selection import train_test_split

print("Loading master annotations...")
df = pd.read_csv('train_annotations.csv')

#Split the DataFrame 80/20
#Random seed 24 to make this reproducible
train_df, val_df = train_test_split(df, test_size=0.2, random_state=24)

#Save
train_df.to_csv('train_split.csv', index=False)
val_df.to_csv('val_split.csv', index=False)

print("--- Data Split Complete ---")
print(f"Total images: {len(df)}")
print(f"Training set (80%): {len(train_df)} images")
print(f"Validation set (20%): {len(val_df)} images")