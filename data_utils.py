'''
This module provides a PyTorch Dataset class for loading images and groundtruths. It also provides a function to extract random 128x128 patches from the 256x256 images in the Dataset
'''
import os
import glob
import random
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import Dataset, TensorDataset, random_split
import torchvision.transforms as transforms

class RoadSkeletonDataset(Dataset):
    def __init__(self, data_dir: str, distort_prob: float = 0.5):
        self.data_dir = data_dir
        self.distort_prob = distort_prob

        # Find and sort image/target paths
        self.image_paths = sorted(glob.glob(os.path.join(data_dir, "image_*.png")))
        self.target_paths = sorted(glob.glob(os.path.join(data_dir, "target_*.png")))

        # Test to insure equal lengths
        assert len(self.image_paths) == len(self.target_paths), \
            "Number of inputs and targets must match"
        
        # Transform to convert PIL->Tensor
        self.to_tensor = transforms.ToTensor()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load images
        img = Image.open(self.image_paths[idx]).convert('L')
        tgt = Image.open(self.target_paths[idx]).convert('L')

        # Apply realistic distortions to the input only
        if random.random() < self.distort_prob:
            # Gaussian blur
            radius = random.uniform(0.5, 1.5)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

            # Additive Gaussian noise
            arr = np.array(img).astype(np.float32) / 255.0
            noise = np.random.normal(loc=0.0, scale=0.05, size=arr.shape).astype(np.float32)
            arr = np.clip(arr + noise, 0.0, 1.0)
            img = Image.fromarray((arr * 255).astype(np.uint8))

        # Convert to tensors (C x H x W)
        img_tensor = self.to_tensor(img)
        tgt_tensor = self.to_tensor(tgt)
        return img_tensor, tgt_tensor


def load_tensor_dataset(data_dir: str, distort_prob: float = 0.5, is_dataset_file: bool = False) -> Dataset:
    '''
    Creates a RoadSkeleton Dataset. Takes in a data directory and distortion probability
    and returns a PyTorch Dataset yielding (input_tensor, target_tensor)
    '''

    if is_dataset_file:
        file_path = data_dir
        dataset = torch.load(file_path, weights_only=False)

        # Apply distortions to the dataset
        if distort_prob > 0.0:
            distorted_dataset = []
            for img, tgt in dataset:
                # Apply realistic distortions to the input only
                if random.random() < distort_prob:

                    img_pil = transforms.ToPILImage()(img)

                    # Gaussian blur
                    radius = random.uniform(0.5, 1.5)
                    img_pil = img_pil.filter(ImageFilter.GaussianBlur(radius=radius))

                    # Additive Gaussian noise
                    arr = np.array(img_pil).astype(np.float32) / 255.0
                    noise = np.random.normal(loc=0.0, scale=0.05, size=arr.shape).astype(np.float32)
                    arr = np.clip(arr + noise, 0.0, 1.0)
                    img_pil = Image.fromarray((arr * 255).astype(np.uint8))

                    img = transforms.ToTensor()(img_pil)

                distorted_dataset.append((img, tgt))

            return distorted_dataset
        
        return dataset
    else:
        return RoadSkeletonDataset(data_dir, distort_prob)

def create_stratified_patches(dataset: RoadSkeletonDataset, pos_patches=6, rand_patches=12, patch_size=128, min_tgt=100):
    '''
    Creates stratified patches of the training dataset. The function extracts a number of positive patches (with a minimum target value) and random patches from the input RoadSkeleton Dataset.
    '''
    imgs, tgts = [], []
    for img, tgt in dataset:
        H, W = img.shape[1:]

        # positive patches
        pos_count = 0
        while pos_count < pos_patches:
            top, left = random.randint(0, H-patch_size), random.randint(0, W-patch_size)
            p_t = tgt[:, top:top+patch_size, left:left+patch_size]
            if p_t.sum() >= min_tgt:
                imgs.append(img[:, top:top+patch_size, left:left+patch_size])
                tgts.append(p_t)
                pos_count += 1
                
        # random patches
        for _ in range(rand_patches):
            top, left = random.randint(0, H-patch_size), random.randint(0, W-patch_size)
            imgs.append(img[:, top:top+patch_size, left:left+patch_size])
            tgts.append(tgt[:, top:top+patch_size, left:left+patch_size])
    return TensorDataset(torch.stack(imgs), torch.stack(tgts))

def create_splits(
    data_dir: str,
    distort_prob: float = 0.5,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    seed: int = None
):
    '''
    Loads the full RoadSkeletonDataset, applies distortions, and splits into train/val/test subsets.
    '''
    # Load full dataset
    full_ds = RoadSkeletonDataset(data_dir, distort_prob)
    total = len(full_ds)

    # Compute split lengths
    train_len = int(total * train_frac)
    val_len = int(total * val_frac)
    test_len = total - train_len - val_len

    # Set seed if provided for reproducibility
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        splits = random_split(full_ds, [train_len, val_len, test_len], generator=generator)
    else:
        splits = random_split(full_ds, [train_len, val_len, test_len])

    return splits  # (train_ds, val_ds, test_ds)
