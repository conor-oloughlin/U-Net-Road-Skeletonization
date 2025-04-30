'''
The following file is used to perform an ablation study on the U-Net model to determine the appropriate configuration for a full training run.
Parameters considered in this study include:
- Learning Rate: [1e-2, 1e-3, 1e-4]
- Loss Functions: [BSE, Dice]
- Architectures: [basic, shallow, deep]
'''

import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from U_Net import UNet
from data_utils import RoadSkeletonDataset, load_tensor_dataset

class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # flatten
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)
        intersection = (probs_flat * targets_flat).sum(1)
        union = probs_flat.sum(1) + targets_flat.sum(1)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * x.size(0)
    
    return running_loss / len(loader.dataset)

def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)
            running_loss += loss.item() * x.size(0)

            # Compute Dice Coefficient
            probs = torch.sigmoid(preds)
            bin_preds = (probs > 0.5).float()
            bin_preds = bin_preds.view(bin_preds.size(0), -1)
            targets = y.view(y.size(0), -1)
            intersection = (bin_preds * targets).sum(1)
            union = bin_preds.sum(1) + targets.sum(1)
            dice = (2. * intersection + 1e-6) / (union + 1e-6)
            dice_scores.append(dice.mean().item())

    avg_loss = running_loss / len(loader.dataset)
    mean_dice = sum(dice_scores) / len(dice_scores)
    return avg_loss, mean_dice

def run_ablation_study():
    # Hyperparameter Grids
    lr_list = [1e-2, 1e-3, 1e-4]
    loss_fns = {
        'BCE': nn.BCEWithLogitsLoss(),
        'Dice': DiceLoss()
    }

    architectures = {
        'basic': [64, 128, 256, 512],
        'shallow': [32, 64, 128, 256],
        'deep': [64, 128, 256, 512, 1024]
    }

    # Non Varying Parameters
    TRAIN_DATA_PATH = 'train_patches_dataset.pth'
    VAL_DATA_PATH = 'val_dataset.pth'
    TEST_DATA_PATH = 'test_dataset.pth'
    BATCH_SIZE = 16
    EPOCHS = 5
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load datasets
    train_dataset = load_tensor_dataset(TRAIN_DATA_PATH, is_dataset_file=True)
    val_dataset = load_tensor_dataset(VAL_DATA_PATH, is_dataset_file=True)
    test_dataset = load_tensor_dataset(TEST_DATA_PATH, is_dataset_file=True)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize results storage
    results = []
    train_hist = {}

    # Run ablation study
    for lr, (loss_name, loss_fn), (arch_name, features) in itertools.product(
        lr_list, loss_fns.items(), architectures.items()):
        config_name = f"lr={lr}_loss={loss_name}_arch={arch_name}"
        print(f"Running {config_name}")

        # Model + optimizer
        model = UNet(in_channels=1, out_channels=1, features=features).to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = loss_fn

        # Short training: 5 epochs
        epoch_loss = []
        for epoch in range(5):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
            epoch_loss.append(train_loss)
        train_hist[config_name] = epoch_loss      

        # Evaluate on validation
        val_loss, dice = evaluate(model, val_loader, criterion, DEVICE)
        results.append({
            'config': config_name,
            'lr': lr,
            'loss': loss_name,
            'arch': arch_name,
            'val_loss': val_loss,
            'dice': dice
        })
    
    # Save results in CSV in main directory in case of error
    if not os.path.exists('ablation_results'):
        os.makedirs('ablation_results')
    
    # Save training history
    train_hist_path = os.path.join('ablation_results', 'backup_train_hist.csv')
    with open(train_hist_path, 'w') as f:
        for config, losses in train_hist.items():
            f.write(f"{config},{','.join(map(str, losses))}\n")
    print(f"Training history saved to {train_hist_path}")
    
    # Save results
    results_path = os.path.join('ablation_results', 'backup_results.csv')
    with open(results_path, 'w') as f:
        f.write("config,lr,loss,arch,val_loss,dice\n")
        for result in results:
            f.write(f"{result['config']},{result['lr']},{result['loss']},{result['arch']},{result['val_loss']},{result['dice']}\n")
    print(f"Results saved to {results_path}")

    
    # Save table of results
    df = pd.DataFrame(results)
    output_dir = './ablation_results'
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join('ablation_results', 'results.csv')
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    # Plot dice coefficient and validation loss
    plt.figure(figsize=(12, 6))
    plt.bar(df['config'], df['dice'])
    plt.xticks(rotation=90)
    plt.ylabel('Dice Coefficient')
    plt.title('Ablation Study: Dice Coefficient on Validation Set')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'val_dice.png'))

    plt.figure(figsize=(12, 6))
    plt.bar(df['config'], df['val_loss'])
    plt.xticks(rotation=90)
    plt.ylabel('Validation Loss')
    plt.title('Ablation Study: Validation Loss')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'val_loss.png'))

    # Plot training loss curves
    plt.figure(figsize=(12, 6))
    for config, losses in train_hist.items():
        plt.plot(range(1, 6), losses, marker='o', label=config)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Ablation Study: Training Loss Curves')
    plt.legend(fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'train_loss_curves.png'))
    print(f"Saved training loss curves to {os.path.join(output_dir, 'train_loss_curves.png')}")

if __name__ == "__main__":
    run_ablation_study()
    print("Ablation study completed.")