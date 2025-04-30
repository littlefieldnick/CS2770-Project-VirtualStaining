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


def evaluate_pix2pix(generator_or_pipeline, val_dataloader, device="cuda", stain="er", multi=False, diffusion=False):
    # Metrics
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    
    if diffusion:
        generator_or_pipeline.unet.eval()
        generator_or_pipeline.class_embedder.eval()
    else:
        generator_or_pipeline.eval()
    
    with torch.no_grad():
        for source, target in tqdm(val_dataloader, desc="Evaluating"):
            source, target = source.to(device), target.to(device)

            if multi and not diffusion:
                # Pix2Pix + Multi
                onehot = torch.zeros(target.shape[0], len(stain_labels), *target.shape[2:], dtype=torch.float32).to(device)
                onehot[:, stain_labels[stain], :, :] = 1.0
                fake_target = generator_or_pipeline(source, onehot)
            elif multi and diffusion:
                # Diffusion + Multi
                class_labels = torch.full((source.shape[0],), stain_labels[stain], device=device, dtype=torch.long)
                fake_target = generator_or_pipeline.forward(
                    source,
                    class_labels,
                    num_inference_steps=50,  # or whatever num steps you want
                )
            else:
                # Pix2Pix + Single
                fake_target = generator_or_pipeline(source)

            # --- Normalize all images to [0,1] before feeding into metrics ---
            target = ((target + 1) / 2).clamp(0, 1)
            fake_target = ((fake_target + 1) / 2).clamp(0, 1)

            # --- Update Metrics ---
            fid.update(target, real=True)
            fid.update(fake_target, real=False)

            ssim.update(fake_target, target)
            psnr.update(fake_target, target)

    # Final compute
    fid_score = fid.compute()
    ssim_score = ssim.compute()
    psnr_score = psnr.compute()
    
    print(f"FID: {fid_score.item():.4f} | SSIM: {ssim_score.item():.4f} | PSNR: {psnr_score.item():.4f}")
    
    return fid_score, ssim_score, psnr_score


# In[4]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[5]:


root_dir = "/ix1/qgu/ngl18/ACROBAT_VirtualStaining_Valis_10x/"
images = os.listdir(root_dir)[15:25]


# In[6]:


def build_dataset(root_dir, ids, real_folder="he", target_folder="er"):
    real = []
    targets = []
    for i in ids:
        if os.path.exists(f"{root_dir}/{i}/{target_folder}"):
            input_images = sorted(list(glob.iglob(f"{root_dir}/{i}/{real_folder}/*.npy")))
            target_images = sorted(list(glob.iglob(f"{root_dir}/{i}/{target_folder}/*.npy")))
            
            real.extend(input_images)
            targets.extend(target_images)

    return real, targets


# In[7]:


from diffusers import DDPMPipeline

class HEClassDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler, class_embedder):
        super().__init__(unet=unet, scheduler=scheduler)
        self.class_embedder = class_embedder

    @torch.no_grad()
    def forward(self, he_images, class_labels, num_inference_steps=100):
        device = he_images.device
        batch_size = he_images.shape[0]
        # 1. Prepare conditioning: class embeddings
        class_labels = class_labels.long()
        class_embeddings = self.class_embedder(class_labels).unsqueeze(1)  # (batch, 1, embed_dim)

        # 2. Prepare latents (start from noise)
        image_shape = (batch_size, self.unet.out_channels, he_images.shape[2], he_images.shape[3])
        latents = torch.randn(image_shape, device=device)

        self.scheduler.set_timesteps(num_inference_steps, device=device)

        for t in tqdm(self.scheduler.timesteps):
            # Concatenate HE + noisy image
            model_input = torch.cat([he_images, latents], dim=1)  # (B, 6, H, W)

            noise_pred = self.unet(
                model_input,
                t,
                encoder_hidden_states=class_embeddings,
            ).sample

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return latents


# In[8]:


valid_source, valid_target = build_dataset(root_dir, images, target_folder="pgr")


# In[10]:


# Example usage:
joint_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
], additional_targets={'target': 'image'})

valid_dataset = VirtualStainingDataset(valid_source, valid_target, transform=joint_transform)
val_dataloader = DataLoader(valid_dataset, batch_size=64, num_workers=16)


# In[11]:


num_classes = 4  # e.g., HER2, ER, Ki67, PGR
embedding_dim = 128  # Size of the class conditioning vector

import torch.nn as nn
import torch.nn.functional as F
# --- Step 1: Define Class Embedder Module ---
class ClassEmbedder(nn.Module):
    def __init__(self, num_classes, embedding_dim):
        super().__init__()
        self.linear = nn.Linear(num_classes, embedding_dim)

    def forward(self, class_labels):
        # One-hot encode class labels
        one_hot = F.one_hot(class_labels, num_classes=num_classes).float()
        # Project to embedding space
        embeddings = self.linear(one_hot)
        return embeddings

# Instantiate the embedder
class_embedder = ClassEmbedder(num_classes=num_classes, embedding_dim=embedding_dim).to(device)


# In[12]:


from diffusers import UNet2DConditionModel, DDPMScheduler, DDIMScheduler

unet = UNet2DConditionModel.from_pretrained("/ix1/qgu/ngl18/VirtualStaining/ckpts/diffusion/unet/").to(device)
# scheduler = DDPMScheduler.from_pretrained("/ix1/qgu/ngl18/VirtualStaining/ckpts/diffusion/scheduler/")
# When creating your scheduler
scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
    beta_schedule="linear",
    clip_sample=False,
    set_alpha_to_one=False,
)
class_embedder.load_state_dict(torch.load("/ix1/qgu/ngl18/VirtualStaining/ckpts/diffusion/class_embedder.pt"))


# In[13]:


pipeline = HEClassDDPMPipeline(unet,scheduler, class_embedder)


# In[ ]:


evaluate_pix2pix(pipeline, val_dataloader, stain="pgr",  multi=True, diffusion=True)


# In[ ]:




