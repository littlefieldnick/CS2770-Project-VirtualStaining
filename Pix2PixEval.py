#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import torch
from sklearn.model_selection import train_test_split
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torch.utils.data import DataLoader
from torchvision import transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Test on H&E -> HER2
from model import Pix2PixGenerator, CascadeNet, HibouIntermediateFeatures
from data import VirtualStainingDataset, build_dataset

from tqdm import tqdm


# In[2]:


stain_labels = {
    "er": 0, 
    "her2": 1,
    "ki67": 2,
    "pgr": 3
}


# In[3]:


def evaluate_pix2pix(generator, val_dataloader, device="cuda", stain="er", multi=False, diffusion=False):
    # Define metrics
    fid = FrechetInceptionDistance(feature=2048, normalize=True)
    psnr = PeakSignalNoiseRatio(data_range=1.0)  # Data range should be 1.0 for normalized images
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    # Move to device
    fid = fid.to(device)
    psnr = psnr.to(device)
    ssim = ssim.to(device)
    
    # Ensure generator is in eval mode
    generator.eval()
    
    with torch.no_grad():
        for source, target in tqdm(val_dataloader):  # A = Input image, B = Ground truth
            source, target = source.to(device), target.to(device)

            if multi and not diffusion:
                onehot = torch.zeros(target.shape[0], len(stain_labels), *target.shape[2:], dtype=torch.float32).to(device)
                onehot[:, stain_labels[stain], :, :] = 1.0
                fake_target = generator(source, onehot).to(device)
            else:
                # Generate fake images
                fake_target = generator(source).to(device)

            # Convert from [-1, 1] to [0, 1] for Inception (FID expects float in [0,255])
            target = ((target + 1) / 2).clamp(0, 1)
            fake_target = ((fake_target + 1) / 2).clamp(0, 1) 

            # Update FID metric
            fid.update(target.float(), real=True)   
            fid.update(fake_target.float(), real=False)  

            # Update SSIM and PSNR metrics
            ssim.update(fake_target, target)  # Fixed variable names
            psnr.update(fake_target, target)  

    # Compute final metrics
    fid_score = fid.compute()
    ssim_score = ssim.compute()
    psnr_score = psnr.compute()
    
    print(f"FID Score: {fid_score.item()}")
    print(f"SSIM Score: {ssim_score.item()}")
    print(f"PSNR Score: {psnr_score.item()}")

    return fid_score, ssim_score, psnr_score  # Return values for logging


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[5]:


root_dir = "/ix1/qgu/ngl18/ACROBAT_VirtualStaining_Valis_10x/"
images = os.listdir(root_dir)[15:25]


# In[6]:


images


# In[7]:


def build_dataset(root_dir, ids, real_folder="he", target_folder="pgr"):
    real = []
    targets = []
    for i in ids:
        if os.path.exists(f"{root_dir}/{i}/{target_folder}"):
            input_images = sorted(list(glob.iglob(f"{root_dir}/{i}/{real_folder}/*.npy")))
            target_images = sorted(list(glob.iglob(f"{root_dir}/{i}/{target_folder}/*.npy")))
            
            real.extend(input_images)
            targets.extend(target_images)

    return real, targets


# In[8]:





# In[9]:


len(valid_source)


# In[10]:


# Example usage:
joint_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
], additional_targets={'target': 'image'})



# In[11]:


import os
os.environ["HF_TOKEN"] = "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"


# In[22]:


# hibou_projection_size = 64
# num_stains = 4
# CG1 = HibouIntermediateFeatures("histai/hibou-b", projection_output=hibou_projection_size)
# CG2 = Pix2PixGenerator(input_channels=3 + hibou_projection_size + num_stains) 

# generator = CascadeNet(CG1, CG2)

# valid_source, valid_target = build_dataset(root_dir, images, target_folder="er")
# valid_dataset = VirtualStainingDataset(valid_source, valid_target, transform=joint_transform)
# val_dataloader = DataLoader(valid_dataset, batch_size=32, num_workers=16)

# ckpt = torch.load(f"/ix1/qgu/ngl18/VirtualStaining/ckpts/cascade2/epoch_200.pth")
# generator.load_state_dict(ckpt["cascade"])
# generator = generator.to(device)
# evaluate_pix2pix(generator, val_dataloader, multi=True, stain="er")


# In[29]:


generator = Pix2PixGenerator(input_channels=3) 

ckpt = torch.load(f"/ix1/qgu/ngl18/VirtualStaining/ckpts/pix2pix2/pgr/epoch_82.pth")
generator.load_state_dict(ckpt["generator"])
generator = generator.to(device)

valid_source, valid_target = build_dataset(root_dir, images, target_folder="pgr")
valid_dataset = VirtualStainingDataset(valid_source, valid_target, transform=joint_transform)
val_dataloader = DataLoader(valid_dataset, batch_size=64, num_workers=16)

evaluate_pix2pix(generator, val_dataloader, stain="pgr")


# In[ ]:





# In[ ]:




