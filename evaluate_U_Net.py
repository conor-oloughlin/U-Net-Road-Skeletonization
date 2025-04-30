import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from scipy.optimize import linear_sum_assignment
from data_utils import load_tensor_dataset
from U_Net import UNet


MODEL_PATH = "unet_model.pth"
TEST_DATA   = "test_dataset.pth"
BATCH_SIZE = 1
DEVICE     = 'mps' if torch.backends.mps.is_available() else 'cpu'

# For bipartite matching
MAX_DIST   = 3.0


def compute_neighbor_counts(mask: np.ndarray) -> np.ndarray:
    '''
    Compute the number of neighbors for each pixel in a binary mask.
    '''

    # Convert to torch tensor
    tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    # Define 8-connected kernel
    kernel = torch.tensor([[1,1,1], [1,0,1], [1,1,1]], dtype=torch.float32)
    kernel = kernel.unsqueeze(0).unsqueeze(0)

    # Convolution
    counts = F.conv2d(tensor, kernel, padding=1)
    return counts.squeeze().numpy().astype(np.int32)


def bipartite_match(gt_pts: np.ndarray, pred_pts: np.ndarray, max_dist: float = MAX_DIST):
    """
    Perform bipartite matching between ground-truth and predicted points.
    Matches only if within max_dist.
    Returns (tp, fp, fn).
    """
    # Both empty
    if len(gt_pts) == 0 and len(pred_pts) == 0:
        return 0, 0, 0
    # Only predictions
    if len(gt_pts) == 0:
        return 0, len(pred_pts), 0
    # Only ground-truths
    if len(pred_pts) == 0:
        return 0, 0, len(gt_pts)

    # Compute distance matrix
    dists = np.linalg.norm(gt_pts[:, None, :] - pred_pts[None, :, :], axis=2)
    cost = dists.copy()
    cost[cost > max_dist] = max_dist + 1
    row_ind, col_ind = linear_sum_assignment(cost)

    # Count true positives, false positive, and false negative
    tp = sum(1 for r, c in zip(row_ind, col_ind) if dists[r, c] <= max_dist)
    fp = len(pred_pts) - tp
    fn = len(gt_pts) - tp
    return tp, fp, fn


def evaluate_model():
    # Device
    dev = torch.device(DEVICE)

    # Load model
    model = UNet(in_channels=1, out_channels=1).to(dev)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=dev))
    model.eval()

    # Test DataLoader
    test_dataset = load_tensor_dataset(TEST_DATA, is_dataset_file=True)
    loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Loss functions
    bce_fn = nn.BCEWithLogitsLoss()
    mse_fn = nn.MSELoss()

    total_bce = 0.0
    total_mse = 0.0
    n_samples = 0
    node_metrics = {v: {'tp':0, 'fp':0, 'fn':0} for v in [1,2,3,4]}

    with torch.no_grad():
        for imgs, tgts in loader:
            imgs, tgts = imgs.to(dev), tgts.to(dev)
            logits = model(imgs)

            # Loss
            bce = bce_fn(logits, tgts)
            probs = torch.sigmoid(logits)
            mse = mse_fn(probs, tgts)
            total_bce += bce.item() * imgs.size(0)
            total_mse += mse.item() * imgs.size(0)
            n_samples += imgs.size(0)

            # Convert to binary masks
            pred_mask = (probs[0,0] > 0.5).cpu().numpy().astype(np.uint8)
            gt_mask   = (tgts[0,0] > 0.5).cpu().numpy().astype(np.uint8)

            # Neighbor counts per pixel (valence)
            pred_counts = compute_neighbor_counts(pred_mask)
            gt_counts   = compute_neighbor_counts(gt_mask)

            # Match nodes per valence
            for v in [1,2,3,4]:
                gt_pts   = np.argwhere(gt_counts == v)
                pred_pts = np.argwhere(pred_counts == v)
                tp, fp, fn = bipartite_match(gt_pts, pred_pts)
                node_metrics[v]['tp'] += tp
                node_metrics[v]['fp'] += fp
                node_metrics[v]['fn'] += fn

    # Final metrics
    test_bce = total_bce / n_samples
    test_mse = total_mse / n_samples
    print(f"Test BCE Loss: {test_bce:.4f}")
    print(f"Test MSE Loss: {test_mse:.4f}\n")
    print("Node Precision & Recall per Valence:")
    for v in [1,2,3,4]:
        m = node_metrics[v]
        tp, fp, fn = m['tp'], m['fp'], m['fn']
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        print(f"  Valence {v}: Precision = {prec:.4f}, Recall = {rec:.4f}")

if __name__ == "__main__":
    evaluate_model()
