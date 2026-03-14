import matplotlib.pyplot as plt
epochs = [1, 2, 3, 4, 5]
# Using the final step loss from each epoch
train_loss = [2.8334, 1.4039, 0.6483, 0.5217, 0.5228] 
val_accuracy = [19.64, 49.17, 66.73, 75.26, 78.88]

fig, ax1 = plt.subplots(figsize=(8, 5))

# Plot Training Loss (Left Y-Axis)
color = 'tab:red'
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
ax1.plot(epochs, train_loss, color=color, marker='o', linewidth=2, label='Training Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xticks(epochs)

# Plot Validation Accuracy (Right Y-Axis)
ax2 = ax1.twinx()  
color = 'tab:blue'
ax2.set_ylabel('Validation Accuracy (%)', color=color, fontweight='bold')
ax2.plot(epochs, val_accuracy, color=color, marker='s', linewidth=2, label='Validation Accuracy')
ax2.tick_params(axis='y', labelcolor=color)

# Formatting
plt.title('ResNet-50: Training Loss and Validation Accuracy', fontweight='bold')
fig.tight_layout()
plt.savefig('learning_curve.png', dpi=300, bbox_inches='tight')
print("Successfully saved learning_curve.png")