import matplotlib.pyplot as plt
from data_utils import load_tensor_dataset
import random
import matplotlib.pyplot as plt
import torch
from data_utils import RoadSkeletonDataset, load_tensor_dataset

def compare_distortion_effects():
    """
    Compare 4 randomly selected images before and after applying distortions.
    Creates a figure showing original image and distorted image for each sample.
    """
    # Data path
    DATA_PATH = 'test_dataset.pth'
    
    # Load dataset with and without distortions
    clean_dataset = load_tensor_dataset(DATA_PATH, distort_prob=0.0, is_dataset_file=True)
    distorted_dataset = load_tensor_dataset(DATA_PATH, distort_prob=0.5, is_dataset_file=True)
    
    # Select 4 random indices
    num_samples = 4
    indices = random.sample(range(len(clean_dataset)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 2 * num_samples))
    
    # Set column titles
    axes[0, 0].set_title("Original Image")
    axes[0, 1].set_title("Distorted Image")
    
    # Set column titles
    axes[0, 0].set_title("Original Image")
    axes[0, 1].set_title("Distorted Image")
    
    # Display the images
    for i, idx in enumerate(indices):
        # Get clean image and corresponding ground truth
        clean_img, ground_truth = clean_dataset[idx]

        # Get the same image with distortions applied
        distorted_img, _ = distorted_dataset[idx]
        
        # Convert tensors to numpy for plotting and squeeze channel dimension
        clean_img_np = clean_img.squeeze().numpy()
        distorted_img_np = distorted_img.squeeze().numpy()

        # Plot original image
        axes[i, 0].imshow(clean_img_np, cmap='gray')
        axes[i, 0].set_ylabel(f"Sample {idx}")
        axes[i, 0].axis('off')
        
        # Plot distorted image
        axes[i, 1].imshow(distorted_img_np, cmap='gray')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_training_patches():
    """
    Display 4 randomly selected patches from the training dataset alongside their ground truths.
    Creates a figure with 4 rows (patches) and 2 columns (input, ground truth).
    """
    # Define data path
    TRAIN_DATA_PATH = 'train_patches_dataset.pth'
    
    # Load the dataset without distortions to see clean patches
    training_dataset = load_tensor_dataset(TRAIN_DATA_PATH, distort_prob=0.0, is_dataset_file=True)
    
    # Pick 4 random indices
    num_samples = 4
    indices = random.sample(range(len(training_dataset)), num_samples)
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2 * num_samples))
    
    # Set column titles
    axes[0, 0].set_title("Input Patch")
    axes[0, 1].set_title("Ground Truth")
    
    # Display the patches
    for i, idx in enumerate(indices):
        # Get patch and corresponding ground truth
        patch, ground_truth = training_dataset[idx]
        
        # Convert tensors to numpy for plotting and squeeze channel dimension
        patch_np = patch.squeeze().numpy()
        ground_truth_np = ground_truth.squeeze().numpy()
        
        # Plot input patch
        axes[i, 0].imshow(patch_np, cmap='gray')
        axes[i, 0].set_ylabel(f"Patch {idx}")
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(ground_truth_np, cmap='gray')
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_distortion_effects()
    visualize_training_patches()