import torch
import matplotlib.pyplot as plt
import random
from U_Net import UNet

# Hard-coded paths
DATASET_PATH = "test_dataset.pth"
MODEL_PATH   = "unet_model.pth"
DEVICE       = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def load_dataset(path):
    """
    Load a dataset saved with torch.save().
    Expects a TensorDataset or list of (img, gt) tuples.
    """
    ds = torch.load(path, weights_only=False)

    # If it's a TensorDataset, convert to list
    if hasattr(ds, "tensors"):
        imgs, gts = ds.tensors
        return [(imgs[i], gts[i]) for i in range(imgs.size(0))]
    return ds

def visualize_samples(dataset, model, device, num_samples=4):
    """
    Randomly sample num_samples examples and display:
    Original, Prediction, Ground Truth in a grid.
    """
    indices = random.sample(range(len(dataset)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples))
    titles = ["Original", "Prediction", "Ground Truth"]
    for ax, t in zip(axes[0], titles):
        ax.set_title(t, fontsize=14)
    
    for row, idx in enumerate(indices):
        img, gt = dataset[idx]
        img_batch = img.unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            logits = model(img_batch)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float().cpu().squeeze()
        
        orig_np = img.cpu().squeeze().numpy()
        gt_np = gt.cpu().squeeze().numpy()
        
        axes[row, 0].imshow(orig_np, cmap="gray")
        axes[row, 1].imshow(pred.numpy(), cmap="gray")
        axes[row, 2].imshow(gt_np, cmap="gray")
        for col in range(3):
            axes[row, col].axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    # Load dataset
    dataset = load_dataset(DATASET_PATH)
    
    # Load model
    model = UNet(in_channels=1, out_channels=1).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()
    
    # Visualize predictions
    visualize_samples(dataset, model, DEVICE, num_samples=4)

if __name__ == "__main__":
    main()