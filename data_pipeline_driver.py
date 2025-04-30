'''
The following file moves through the datapipeline. Running the file creates a custom dataset of thick images and thin groundtruths and splits them into testing, training, and validation sets.
128x128 patches are then extracted from the training dataset and saved for model training.
'''
import torch
from data_utils import RoadSkeletonDataset, create_stratified_patches, create_splits, load_tensor_dataset

def run_data_pipeline():
    # Set parameters
    DATA_DIR = 'thinning_data/data/thinning'
    DISTORTION_PROB = 0.5
    SEED = 2025
    POSITIVE_PATCHES = 6
    RANDOM_PATCHES = 12
    PATCH_SIZE = 128
    MIN_TARGET = 100

    # split into train/val/test
    splits = create_splits(DATA_DIR, distort_prob=DISTORTION_PROB, seed=SEED)
    train_dataset, val_dataset, test_dataset = splits

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    # Create stratified patches for training
    train_patches = create_stratified_patches(train_dataset, pos_patches=POSITIVE_PATCHES, rand_patches=RANDOM_PATCHES, patch_size=PATCH_SIZE, min_tgt=MIN_TARGET)
    print(f"Train patches size: {len(train_patches)}")

    # Save train patches, val dataset, and test dataset
    torch.save(train_patches, 'train_patches_dataset.pth')
    torch.save(val_dataset, 'val_dataset.pth')
    torch.save(test_dataset, 'test_dataset.pth')
    print("Train patches, val dataset, and test dataset saved.")

if __name__ == "__main__":
    run_data_pipeline()
    print("Data pipeline completed.")
