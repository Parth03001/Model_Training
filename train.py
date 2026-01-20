import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from transformers import AutoImageProcessor, AutoModelForImageClassification
from transformers import EfficientNetImageProcessor, EfficientNetForImageClassification

from tqdm import tqdm
import numpy as np
import json

class ConnectorDataset(Dataset):
    def __init__(self, image_paths, labels, processor, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.augment = augment
        
        if augment:
            self.transform = transforms.Compose([
                transforms.RandomRotation(15),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            ])
        else:
            self.transform = None
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        return {
            'pixel_values': pixel_values,
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def load_data(data_dir):
    """Load images from ok and not_ok folders"""
    data_dir = Path(data_dir)
    ok_dir = data_dir / 'ok'
    not_ok_dir = data_dir / 'not_ok'
    
    image_paths = []
    labels = []
    
    # Load OK images (label = 0)
    if ok_dir.exists():
        for img_path in ok_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_paths.append(str(img_path))
                labels.append(0)
    
    # Load NOT OK images (label = 1)
    if not_ok_dir.exists():
        for img_path in not_ok_dir.glob('*'):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                image_paths.append(str(img_path))
                labels.append(1)
    
    print(f"Loaded {len([l for l in labels if l == 0])} OK images")
    print(f"Loaded {len([l for l in labels if l == 1])} NOT OK images")
    
    return image_paths, labels

def train_model(data_dir, model_name='microsoft/resnet-50', epochs=20, batch_size=16, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    image_paths, labels = load_data(data_dir)
    
    # Split data: 70% train, 15% val, 15% test
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        image_paths, labels, test_size=0.3, random_state=42, stratify=labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
    )
    
    print(f"\nData split:")
    print(f"Train: {len(train_paths)} images")
    print(f"Val: {len(val_paths)} images")
    print(f"Test: {len(test_paths)} images")
    
    # Save test data for later evaluation
    test_data = {
        'paths': test_paths,
        'labels': test_labels
    }
    with open('test_data.json', 'w') as f:
        json.dump(test_data, f)
    
    # Load processor and model
    processor = EfficientNetImageProcessor.from_pretrained(model_name)
    model = EfficientNetForImageClassification.from_pretrained(
        model_name,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    model.to(device)
    
    # Create datasets
    train_dataset = ConnectorDataset(train_paths, train_labels, processor, augment=True)
    val_dataset = ConnectorDataset(val_paths, val_labels, processor, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_acc = 0.0
    patience = 5
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]")
        for batch in pbar:
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values=pixel_values)
            loss = criterion(outputs.logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(pixel_values=pixel_values)
                loss = criterion(outputs.logits, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
            model.save_pretrained('best_model_hf')
            processor.save_pretrained('best_model_hf')
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    print(f"\n✓ Training complete! Best Val Acc: {best_val_acc:.2f}%")
    return model, processor

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to data directory with ok/not_ok folders')
    parser.add_argument('--model', type=str, default='microsoft/resnet-50', help='Model name from HuggingFace')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    
    args = parser.parse_args()
    
    train_model(args.data_dir, args.model, args.epochs, args.batch_size, args.lr)