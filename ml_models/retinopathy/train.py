import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from PIL import Image
import seaborn as sns

# Set seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check for GPU acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Paths
base_dir = r"C:\Study\PN\python medical project\mediscan"
csv_path = os.path.join(base_dir, "datasets", "retinopathy", "train.csv")
images_dir = os.path.join(base_dir, "datasets", "retinopathy", "train_images")
model_save_dir = os.path.join(base_dir, "ml_models", "retinopathy")
model_save_path = os.path.join(model_save_dir, "retinopathy_model.pth")
class_indices_path = os.path.join(model_save_dir, "class_indices.json")
cm_plot_path = os.path.join(model_save_dir, "confusion_matrix.png")
history_plot_path = os.path.join(model_save_dir, "training_history.png")

os.makedirs(model_save_dir, exist_ok=True)

# 1. DATA PREPARATION & CUSTOM DATASET
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"Annotations CSV file not found at: {csv_path}")

df = pd.read_csv(csv_path)

# Map diagnosis to binary label (0 = No DR, 1-4 = Has DR)
df['label'] = (df['diagnosis'] > 0).astype(int)

# Ensure filenames have correct extension (.png)
df['filename'] = df['id_code'].apply(lambda x: f"{x}.png" if not str(x).endswith('.png') else str(x))

# 80/20 train/validation stratified split
train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df['label'],
    random_state=42
)

# Custom Dataset Class
class RetinopathyDataset(Dataset):
    def __init__(self, dataframe, images_directory, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.images_dir = images_directory
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, row['filename'])
        image = Image.open(img_path).convert('RGB')
        label = int(row['label'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

# Transform specs
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(20),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.1), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create datasets & loaders
train_dataset = RetinopathyDataset(train_df, images_dir, transform=train_transform)
val_dataset = RetinopathyDataset(val_df, images_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Save class indices mapping
class_to_idx = {"0": 0, "1": 1}
with open(class_indices_path, 'w', encoding='utf-8') as f:
    json.dump(class_to_idx, f, indent=4)
print(f"Saved class indices mapping to {class_indices_path}")

# 2. CLASS IMBALANCE HANDLING
y_train = train_df['label'].values
classes = np.unique(y_train)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)
class_weight_dict = dict(zip(classes, class_weights))
print(f"Computed class weights: {class_weight_dict}")

# 3. MODEL ARCHITECTURE (MobileNetV2 Transfer Learning)
class RetinopathyModel(nn.Module):
    def __init__(self):
        super(RetinopathyModel, self).__init__()
        self.base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        for param in self.base.parameters():
            param.requires_grad = False
            
        num_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.base(x)

model = RetinopathyModel().to(device)
optimizer = torch.optim.Adam(model.base.classifier.parameters(), lr=0.001)

# Training loop
epochs = 20
patience = 5
patience_counter = 0
best_acc = 0.0
best_loss = float('inf')

history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

print("Starting PyTorch training with class weights...")
for epoch in range(epochs):
    # Train epoch
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device).float().unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate sample-specific weights dynamically for class balancing (matching Keras class_weight)
        batch_weights = torch.zeros_like(labels)
        batch_weights[labels == 0] = class_weight_dict[0]
        batch_weights[labels == 1] = class_weight_dict[1]
        
        loss = nn.functional.binary_cross_entropy(outputs, labels, weight=batch_weights)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        preds = (outputs > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total
    
    # Validation epoch (no class weights applied on validation loss evaluation)
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device).float().unsqueeze(1)
            
            outputs = model(images)
            loss = nn.functional.binary_cross_entropy(outputs, labels)
            
            val_loss += loss.item() * images.size(0)
            preds = (outputs > 0.5).float()
            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)
            
    epoch_val_loss = val_loss / len(val_dataset)
    epoch_val_acc = val_correct / val_total
    
    print(f"Epoch {epoch+1:02d}/{epochs} - Loss: {epoch_loss:.4f} - Acc: {epoch_acc:.4f} - Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}")
    
    # Save metrics
    history['loss'].append(epoch_loss)
    history['accuracy'].append(epoch_acc)
    history['val_loss'].append(epoch_val_loss)
    history['val_accuracy'].append(epoch_val_acc)
    
    # Checkpoint (Save if val accuracy improves)
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        torch.save(model.state_dict(), model_save_path)
        print(f"--> Saved best model with validation accuracy: {best_acc:.4f}")
        
    # Early stopping check (based on validation loss)
    if epoch_val_loss < best_loss:
        best_loss = epoch_val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

# 4. EVALUATION & METRICS GENERATION
if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path))
model.eval()

all_preds = []
all_trues = []

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        preds = (outputs > 0.5).cpu().numpy().astype(int).flatten()
        all_preds.extend(preds)
        all_trues.extend(labels.numpy())

print("\n=========================================")
print("CLASSIFICATION REPORT")
print("=========================================")
class_names = ['No DR', 'Has DR']
report = classification_report(
    all_trues,
    all_preds,
    target_names=class_names
)
print(report)

print(f"Final Test Accuracy: {best_acc * 100:.2f}%")

# Confusion Matrix Chart
cm = confusion_matrix(all_trues, all_preds)
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title('Confusion Matrix - Retinopathy Classification')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(cm_plot_path)
plt.close()
print(f"Confusion matrix plot saved to {cm_plot_path}")

# Training History Graphs
plt.figure(figsize=(12, 4))

# Accuracy curve
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy', color='#00d4ff', linewidth=2)
plt.plot(history['val_accuracy'], label='Val Accuracy', color='#ff4d6d', linewidth=2)
plt.title('Model Accuracy History')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

# Loss curve
plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss', color='#00d4ff', linewidth=2)
plt.plot(history['val_loss'], label='Val Loss', color='#ff4d6d', linewidth=2)
plt.title('Model Loss History')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.savefig(history_plot_path)
plt.close()
print(f"Training history plot saved to {history_plot_path}")

# Save training history as JSON
history_data = {
    'epochs': list(range(1, len(history['accuracy']) + 1)),
    'accuracy': history['accuracy'],
    'val_accuracy': history['val_accuracy'],
    'loss': history['loss'],
    'val_loss': history['val_loss'],
    'train_acc': history['accuracy'],
    'val_acc': history['val_accuracy']
}
with open(os.path.join(model_save_dir, 'history.json'), 'w') as f:
    json.dump(history_data, f)
print(f"Training history JSON saved to {os.path.join(model_save_dir, 'history.json')}")

# Save evaluation metrics as JSON
from sklearn.metrics import precision_score, recall_score, f1_score
metrics_data = {
    'accuracy': float(best_acc),
    'precision': float(precision_score(all_trues, all_preds, average='weighted')),
    'recall': float(recall_score(all_trues, all_preds, average='weighted')),
    'f1': float(f1_score(all_trues, all_preds, average='weighted'))
}
with open(os.path.join(model_save_dir, 'metrics.json'), 'w') as f:
    json.dump(metrics_data, f)
print(f"Evaluation metrics JSON saved to {os.path.join(model_save_dir, 'metrics.json')}")

