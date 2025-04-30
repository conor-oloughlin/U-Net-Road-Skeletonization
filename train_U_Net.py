import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from data_utils import create_splits
from U_Net import UNet
from data_utils import RoadSkeletonDataset, load_tensor_dataset

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Compute Pixel Accuracy
        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()
        correct += (preds == y).sum().item()
        total += y.numel()
    
    train_accuracy = correct / total
    avg_loss = running_loss / len(loader)
    return avg_loss, train_accuracy

def validate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    dice_scores = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)

            running_loss += loss.item()

            # Compute Pixel Accuracy
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()
            correct += (preds == y).sum().item()
            total += y.numel()

            # Compute Dice Coefficient
            preds = preds.view(preds.size(0), -1)
            targets = y.view(y.size(0), -1)
            intersection = (preds * targets).sum(1)
            union = preds.sum(1) + targets.sum(1)
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            dice_scores.append(dice.mean().item())
    
    val_accuracy = correct / total
    mean_dice = sum(dice_scores) / len(dice_scores)
    avg_loss = running_loss / len(loader)

    return avg_loss, val_accuracy, mean_dice

def main():
    # Hyperparameters
    TRAIN_DATA_PATH = 'train_patches_dataset.pth'
    VAL_DATA_PATH = 'val_dataset.pth'
    TEST_DATA_PATH = 'test_dataset.pth'
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 0.001
    DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')

    # Load Datasets
    train_dataset = load_tensor_dataset(TRAIN_DATA_PATH, is_dataset_file=True)
    val_dataset = load_tensor_dataset(VAL_DATA_PATH, is_dataset_file=True)
    test_dataset = load_tensor_dataset(TEST_DATA_PATH, is_dataset_file=True)

    # Convert to DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Training loop
    early_stop_patience = 20
    best_dice = 0.0
    epochs_since_improve = 0
    for epoch in range(EPOCHS):
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_accuracy, val_dice = validate_epoch(model, val_loader, criterion, DEVICE)
        scheduler.step(train_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] - "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Val Dice: {val_dice:.4f}")

        # Early stopping
        if val_dice > best_dice:
            best_dice = val_dice
            epochs_since_improve = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            epochs_since_improve += 1

            if epochs_since_improve >= early_stop_patience:
                print('Stopping early at epoch:', epoch + 1)
                break
    
    print("Training complete.")
    
    # Evaluate on Test Set
    test_loss, test_accuracy, test_dice = validate_epoch(model, test_loader, criterion, DEVICE)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test Dice: {test_dice:.4f}")

    # Save the final model
    torch.save(model.state_dict(), 'unet_model.pth')
    print("Model saved.")

if __name__ == "__main__":
    main()


