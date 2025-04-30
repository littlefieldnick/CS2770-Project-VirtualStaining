#!/usr/bin/env python
# coding: utf-8

# In[47]:


import numpy as np
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from data import VirtualStainingDataset, build_dataset

from pytorch_msssim import ssim

import albumentations as A
from albumentations.pytorch import ToTensorV2

# In[3]:


from sklearn.model_selection import train_test_split


# In[19]:


root_dir = "/ix1/qgu/ngl18/ACROBAT_VirtualStaining_Valis_10x/"


# In[66]:


images = os.listdir(root_dir)[:15]
stain = "ki67"


# In[69]:


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


# In[70]:


train_dataset_real, train_dataset_target = build_dataset(root_dir, images, target_folder=stain)

# In[71]:


len(train_dataset_real)


# In[72]:


import torch
import torch.nn as nn

class Pix2PixGenerator(nn.Module):
    def __init__(self, input_channels=3):
        super(Pix2PixGenerator, self).__init__()

        # Encoder Layers: 64 -> 128 -> 256 -> 512 -> 512 -> 512 -> 512 -> 512
        self.enc1 = self.conv_block(input_channels, 64)    
        self.enc2 = self.conv_block(64, 128)  
        self.enc3 = self.conv_block(128, 256) 
        self.enc4 = self.conv_block(256, 512) 
        self.enc5 = self.conv_block(512, 512) 
        self.enc6 = self.conv_block(512, 512) 
        self.enc7 = self.conv_block(512, 512) 

        # U-Net Decoder layers 512 -> 1024 -> 1024 -> 1024 -> 1024 -> 512 -> 256 -> 128
        self.up1 = self.deconv_block(512, 512)   
        self.up2 = self.deconv_block(1024, 1024)  # Fix the input channels here (1024 + 512)
        self.up3 = self.deconv_block(1024 + 512, 1024)  # Fix the input channels here (1024 + 256)
        self.up4 = self.deconv_block(1024 + 512, 1024)  # Fix the input channels here (1024 + 128)
        self.up5 = self.deconv_block(1024 + 256, 512)    # Fix the input channels here (1024 + 64)
        self.up6 = self.deconv_block(512 + 128, 256)     # Fix the input channels here (512 + 64)
        self.up7 = self.deconv_block(256 + 64, 128)     # Fix the input channels here (256 + 64)

        # Final output layer (RGB output)
        self.final_conv = nn.Conv2d(128, 3, kernel_size=3, stride=1, padding=1)
        self.tanh = nn.Tanh()
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def deconv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        enc5_out = self.enc5(enc4_out)
        enc6_out = self.enc6(enc5_out)
        enc7_out = self.enc7(enc6_out)

        # Decoder with skip connections
        up1_out = self.up1(enc7_out)  # This will be 512 channels
        up2_out = self.up2(torch.cat([up1_out, enc6_out], 1))  # Concatenate skip connection (1024 + 512 = 1536 channels)
        up3_out = self.up3(torch.cat([up2_out, enc5_out], 1))  # Concatenate skip connection (1024 + 256 = 1280 channels)
        up4_out = self.up4(torch.cat([up3_out, enc4_out], 1))  # Concatenate skip connection (1024 + 128 = 1152 channels)
        up5_out = self.up5(torch.cat([up4_out, enc3_out], 1))  # Concatenate skip connection (1024 + 64 = 1088 channels)
        up6_out = self.up6(torch.cat([up5_out, enc2_out], 1))  # Concatenate skip connection (512 + 64 = 576 channels)
        up7_out = self.up7(torch.cat([up6_out, enc1_out], 1))  # Concatenate skip connection (256 + 64 = 320 channels)

        # Final output layer
        output = self.final_conv(up7_out)  # Output with 3 channels (RGB)

        return self.tanh(output)


# In[73]:


import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(PatchGANDiscriminator, self).__init__()
        
        # Define layers for PatchGAN discriminator
        self.model = nn.Sequential(
            # Layer 1 
            nn.Conv2d(input_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 2 
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 3 
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Layer 4 
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Final output layer: Convolution that outputs a single channel (real or fake)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)
        )

    def forward(self, x):
        return self.model(x)

# Example usage:
# Assuming input_channels is 3 (RGB images)
discriminator = PatchGANDiscriminator(input_channels=3)


# In[74]:


import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# Example usage:
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.Affine(scale=(0.98, 1.02), translate_percent=0.05, rotate=(-5, 5), p=0.7),
    A.ElasticTransform(alpha=1, sigma=50, alpha_affine=5, p=0.2),
    A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.7),
    A.RandomGamma(p=0.3),
    # A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ToTensorV2()
], additional_targets={'target': 'image'})


# Create dataset instance
# Create DataLoader instance
train_dataset = VirtualStainingDataset(train_dataset_real, train_dataset_target, transform=train_transform) 

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)


# In[75]:


import torchvision.models as models
import torch.nn as nn
from torchvision import transforms

class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True).features[:16]  # Up to relu_2_2
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.resize = resize

        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # VGG expects these
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, pred, target):
        # Assume input is [-1, 1] → convert to [0, 1]
        pred = (pred + 1) / 2
        target = (target + 1) / 2

        # Resize if needed
        if self.resize:
            pred = F.interpolate(pred, size=(224, 224), mode='bilinear', align_corners=False)
            target = F.interpolate(target, size=(224, 224), mode='bilinear', align_corners=False)

        # Normalize for VGG
        pred = self.normalize(pred)
        target = self.normalize(target)

        pred_feat = self.vgg(pred)
        target_feat = self.vgg(target)

        return F.l1_loss(pred_feat, target_feat)


# In[ ]:


import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Assuming you have your generator, discriminator, and dataset
# Define the generator and discriminator
generator = Pix2PixGenerator() # Replace with your generator model
discriminator = PatchGANDiscriminator()  # Assuming RGB images

# Setup for device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator.to(device)
discriminator.to(device)

# Loss function
adversarial_loss = nn.MSELoss()  # Adversarial loss (mean squared error)
pixelwise_loss = nn.L1Loss()  # Pixel-wise loss (L1 loss)
percep_loss_fn = VGGPerceptualLoss().to(device)

# Optimizers for both networks
lr_G = 0.0002  # Learning rate'
lr_D = 0.0001

beta1 = 0.5  # Beta1 for Adam optimizer
lambda_l1 = 10
lambda_ssim = 1       # Stronger emphasis on shape/structure
lambda_percep = 0.5     # Prioritize realism and stain texture

def linear_decay_lambda(epoch, start_decay_epoch=75, total_epochs=150):
    if epoch < start_decay_epoch:
        return 1.0
    else:
        return 1.0 - (epoch - start_decay_epoch) / (total_epochs - start_decay_epoch)


num_epochs = 200
optimizer_G = optim.Adam(generator.parameters(), lr=lr_G, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_D, betas=(beta1, 0.999))

from torch.optim.lr_scheduler import LambdaLR

scheduler_G = LambdaLR(optimizer_G, lr_lambda=lambda epoch: linear_decay_lambda(epoch, 100, 200))

# Training loop
for epoch in range(num_epochs + 1):
    generator.train()
    discriminator.train()

    for i, (real_images, target_images) in enumerate(train_loader):
        real_images = real_images.to(device)
        target_images = target_images.to(device)

        # ------------------
        #  Train Discriminator
        # ------------------

        optimizer_D.zero_grad()

        # Generate fake images using the generator
        fake_images = generator(real_images)
        
        # Discriminator loss on real images
        real_preds = discriminator(target_images)

        # Real images labels (1s) and fake images labels (0s)
        real_labels = torch.ones_like(real_preds).to(device)
        real_loss = adversarial_loss(real_preds, real_labels)

        # Discriminator loss on fake images
        fake_preds = discriminator(fake_images.detach())  # Detach to avoid backprop through generator
        fake_labels = torch.zeros_like(fake_preds).to(device)

        fake_loss = adversarial_loss(fake_preds, fake_labels)

        # Total discriminator loss
        d_loss = (real_loss + fake_loss) / 2

        # Backprop and optimize discriminator
        d_loss.backward()
        optimizer_D.step()

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Generator loss (adversarial loss + pixel-wise loss)
        fake_preds = discriminator(fake_images)
        g_adversarial_loss = adversarial_loss(fake_preds, real_labels)
        g_pixelwise_loss = pixelwise_loss(fake_images, target_images)

        # SSIM loss (we subtract SSIM to make it a minimization objective)
        g_ssim_loss = 1 - ssim(fake_images, target_images, data_range=2.0, size_average=True)
        loss_percep = percep_loss_fn(fake_images, target_images)
        g_loss = g_adversarial_loss + lambda_l1 * g_pixelwise_loss + lambda_ssim * g_ssim_loss + lambda_percep * loss_percep

        # Backprop and optimize generator
        g_loss.backward()
        optimizer_G.step()

        # Print training stats
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}] "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Optional: visualization every few epochs
    if (epoch + 1) % 2 == 0:
        generator.eval()
        with torch.no_grad():
            he_sample = real_images[:4] if real_images.shape[0] > 4 else real_images
            ihc_sample = target_images[:4]  if target_images.shape[0] > 4 else target_images
           
            ihc_pred = generator(he_sample)
        
            def to_img(x): return (x * 0.5 + 0.5).clamp(0, 1)
        
            fig, axs = plt.subplots(3, he_sample.shape[0], figsize=(12, 9))
            for j in range(he_sample.shape[0]):
                axs[0, j].imshow(to_img(he_sample[j].permute(1, 2, 0)).cpu())
                axs[0, j].set_title("H&E")    
                axs[1, j].imshow(to_img(ihc_sample[j].permute(1, 2, 0)).cpu())
                axs[1, j].set_title("Real IHC")
                axs[2, j].set_title("Pred IHC")
                axs[2, j].imshow(to_img(ihc_pred[j].permute(1, 2, 0)).cpu())
                for row in axs: row[j].axis('off')
            plt.tight_layout()
            plt.savefig(f"/ix1/qgu/ngl18/VirtualStaining/outputs/pix2pix2/{stain}/he_{stain}_{epoch+1}.png")

        torch.save({
            "generator": generator.state_dict(),
            "discriminator": discriminator.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
            "scheduler_G": scheduler_G.state_dict(),
        }, f"/ix1/qgu/ngl18/VirtualStaining/ckpts/pix2pix2/{stain}/epoch_{epoch+1}.pth")
       
    scheduler_G.step()


# In[ ]:





# In[ ]:




