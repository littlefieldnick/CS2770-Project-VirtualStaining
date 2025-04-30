#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import HibouIntermediateFeatures, Pix2PixGenerator, PatchGANDiscriminator, CascadeNet
from data import VirtualStainingDataset, build_dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoModel, AutoConfig

from pytorch_msssim import ssim

import matplotlib.pyplot as plt
from torchmetrics.segmentation import DiceScore

from skimage.color import rgb2hed
from skimage.filters import threshold_otsu
from skimage.color import separate_stains, hed_from_rgb


# In[2]:


class MultiStainVirtualStainingDataset(Dataset):
    def __init__(self, input_images, stain_ids_dict, stain_label_dict, num_stains, transform=None):
        """
        Args:
            input_images (list): Paths to H&E tiles. Assumes corresponding IHC tile has same file name!
            stain_ids_dict (dict): Maps idx to list of stain label ids (e.g., [0, 1, 3]).
            num_stains (int): Total number of stain types (for one-hot encoding).
            transform (callable, optional): Albumentations transform.
        """
        self.input_images = input_images
        self.stain_ids_dict = stain_ids_dict
        self.stain_label_2_id = stain_label_dict
        self.id_2_stain_label = {sid: stain for stain, sid in self.stain_label_2_id.items()}
        self.num_stains = num_stains
        self.transform = transform

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        # Load H&E image
        input_image = np.load(self.input_images[idx])
        if input_image.shape[-1] == 4:
            input_image = input_image[:, :, :3]

        case_num = self.input_images[idx].split("/")[-1].split("_")[0]

        # Randomly select a stain for this sample
        stain_label = self.stain_ids_dict[case_num]
        stain_idx = np.random.choice(stain_label, size=1)[0]

        target_path = self.input_images[idx].replace("he", self.id_2_stain_label[stain_idx])

        # Load selected IHC image
        target_image = np.load(target_path)
        if target_image.shape[-1] == 4:
            target_image = target_image[:, :, :3]

        # Apply transforms
        if self.transform:
            augmented = self.transform(image=input_image, target=target_image)
            input_image = augmented['image']
            target_image = augmented['target']
        else:
            input_image = torch.tensor(input_image.transpose(2, 0, 1), dtype=torch.float32) / 255.
            target_image = torch.tensor(target_image.transpose(2, 0, 1), dtype=torch.float32) / 255.

        # One-hot encode stain
        one_hot = torch.zeros(self.num_stains, *input_image.shape[1:], dtype=torch.float32)
        one_hot[stain_label] = 1.0

        return input_image, one_hot, target_image


# In[3]:


stain_labels = {
    "er": 0, 
    "her2": 1,
    "ki67": 2,
    "pgr": 3
}


# In[11]:


root_dir = "/ix1/qgu/ngl18/ACROBAT_VirtualStaining_Valis_10x"
images = os.listdir(root_dir)[:15]

def build_dataset(root_dir, ids, stain_labels, real_folder="he"):
    real = []
    targets = dict()
    for i in ids:
        input_images = sorted(list(glob.iglob(f"{root_dir}/{i}/{real_folder}/*.npy")))
        target_stains = list(os.listdir(f"{root_dir}/{i}/"))
        
        real.extend(input_images)
        targets[i] = [stain_labels[target] for target in target_stains if target != real_folder]

    return real, targets

import albumentations as A
from albumentations.pytorch import ToTensorV2

joint_transform = A.Compose([
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
    
train_dataset_real, train_dataset_target = build_dataset(root_dir, images, stain_labels)


# In[12]:


train_dataset_target


# In[13]:


dataset = MultiStainVirtualStainingDataset(train_dataset_real, train_dataset_target, stain_labels, 4, transform=joint_transform) 


# In[14]:


train_loader = DataLoader(dataset, batch_size=32, num_workers=16, shuffle=True)


# In[15]:


class HibouIntermediateFeatures(nn.Module):
    def __init__(self, backbone_name="histai/hibou-b", projection_output=32, upsample_size=(256, 256)):
        super().__init__()
        self.hibou_model = AutoModel.from_pretrained(backbone_name, trust_remote_code=True)
        self.hibou_model.requires_grad_(False)  # freeze Hibou
        self.project = nn.Conv2d(768, projection_output, kernel_size=1)
        self.upsample_size = upsample_size

    def forward(self, x):
        with torch.no_grad():  # freeze Hibou backbone
            x_rep = self.hibou_model(x, return_dict=True)
            tokens = x_rep.last_hidden_state  # shape: [B, N, C]

        tokens = tokens[:, 1:257, :]  # (B, N, 768)
        B, N, C = tokens.shape
        H = W = int(N ** 0.5)
        if H * W != N:
            raise ValueError(f"Token count {N} doesn't match (H x W = {H} x {W})")

        # Reshape to spatial format
        feat_map = tokens.permute(0, 2, 1).reshape(B, C, H, W)

        # Project to desired channels
        projected = self.project(feat_map)

        # Upsample to match image resolution
        upsampled = F.interpolate(projected, size=self.upsample_size, mode="bilinear", align_corners=False)

        return upsampled


class CascadeNet(nn.Module):
    def __init__(self, hibou_model: nn.Module, generator: nn.Module):
        super().__init__()
        self.hibou = hibou_model  # frozen or trainable
        self.generator = generator  # e.g., Pix2PixGenerator(input_channels=3 + hibou_channels)

    def forward(self, x, stain_map):
        # x: input H&E tile [B, 3, H, W]
        hibou_feat = self.hibou(x)  # [B, hibou_channels, H, W], upsampled already

        # Concatenate original input with features
        concat = torch.cat([x, hibou_feat, stain_map], dim=1)  # [B, 3 + hibou_channels, H, W]

        # Generate final image
        return self.generator(concat)

    
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


# In[16]:


hibou_projection_size = 64
num_stains = 4
CG1 = HibouIntermediateFeatures("histai/hibou-b", projection_output=hibou_projection_size)
CG2 = Pix2PixGenerator(input_channels=3 + hibou_projection_size + num_stains) 
D2 = PatchGANDiscriminator()  # Assuming RGB images

cascade_net = CascadeNet(CG1, CG2)


# In[17]:


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


# Setup for device
import torch.nn.functional as F
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cascade_net.to(device)
D2.to(device)

# Loss function
adversarial_loss = nn.MSELoss()  # Adversarial loss (mean squared error)
pixelwise_loss = nn.L1Loss()  # Pixel-wise loss (L1 loss)
percep_loss_fn = VGGPerceptualLoss().to(device)

# Optimizers for both networks
lr_G = 2e-04  # Learning rate'
lr_D = 1e-04

beta1 = 0.5  # Beta1 for Adam optimizer

def linear_decay_lambda(epoch, start_decay_epoch=10, total_epochs=20):
    if epoch < start_decay_epoch:
        return 1.0
    else:
        return 1.0 - (epoch - start_decay_epoch) / (total_epochs - start_decay_epoch)


num_epochs = 200
optimizer_G = optim.Adam(cascade_net.parameters(), lr=lr_G, betas=(beta1, 0.999))
optimizer_D = optim.Adam(D2.parameters(), lr=lr_D, betas=(beta1, 0.999))

# scheduler_G = lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs, eta_min=0)
# scheduler_D = lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=num_epochs, eta_min=0)
from torch.optim.lr_scheduler import LambdaLR

scheduler_G = LambdaLR(optimizer_G, lr_lambda=lambda epoch: linear_decay_lambda(epoch, 100, 200))
# scheduler_D = LambdaLR(optimizer_D, lr_lambda=lambda epoch: linear_decay_lambda(epoch, 10, 20))

# Training loop
for epoch in range(num_epochs + 1):
    cascade_net.train()
    D2.train()

    for i, (real_images, one_hot_stain, target_images) in enumerate(train_loader):
        real_images = real_images.to(device)
        one_hot_stain = one_hot_stain.to(device)
        target_images = target_images.to(device)

        # ------------------
        #  Train Discriminator
        # ------------------

        optimizer_D.zero_grad()

        # Generate fake images using the generator
        fake_images = cascade_net(real_images, one_hot_stain)
        
        # Discriminator loss on real images
        real_preds = D2(target_images)

        # Real images labels (1s) and fake images labels (0s)
        real_labels = torch.ones_like(real_preds).to(device)
        real_loss = adversarial_loss(real_preds, real_labels)

        # Discriminator loss on fake images
        fake_preds = D2(fake_images.detach())  # Detach to avoid backprop through generator
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
        
        # Generator loss (adversarial loss + pixel-wise loss + SSIM)
        fake_preds = D2(fake_images)
        g_adversarial_loss = adversarial_loss(fake_preds, real_labels)
        
        # Standard L1 loss
        g_pixelwise_loss = pixelwise_loss(fake_images, target_images)
        
        # SSIM loss (we subtract SSIM to make it a minimization objective)
        g_ssim_loss = 1 - ssim(fake_images, target_images, data_range=2.0, size_average=True)
        loss_percep = percep_loss_fn(fake_images, target_images)

        # Combined generator loss
        lambda_l1 = 10        # Mild structure preservation
        lambda_ssim = 1       # Stronger emphasis on shape/structure
        lambda_percep = 0.5     # Prioritize realism and stain texture

        
        # Combine them
        g_loss = g_adversarial_loss + lambda_l1 * g_pixelwise_loss + lambda_ssim * g_ssim_loss + lambda_percep * loss_percep
        
        # Backprop and optimize generator
        g_loss.backward()
        optimizer_G.step()
        # Print training stats
        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}] "
                  f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")

    # Optional: visualization every few epochs
    if (epoch + 1) % 10 == 0:
        cascade_net.eval()
        with torch.no_grad():
            he_sample = real_images[:4]
            ihc_sample = target_images[:4]
           
            ihc_pred = cascade_net(he_sample, one_hot_stain[:4])

            def to_img(x): return (x * 0.5 + 0.5).clamp(0, 1)

            fig, axs = plt.subplots(3, 4, figsize=(12, 9))
            for j in range(4):
                axs[0, j].imshow(to_img(he_sample[j].permute(1, 2, 0)).cpu())
                axs[0, j].set_title("H&E")    
                axs[1, j].imshow(to_img(ihc_sample[j].permute(1, 2, 0)).cpu())
                axs[1, j].set_title("Real IHC")
                axs[2, j].set_title("Pred IHC")
                axs[2, j].imshow(to_img(ihc_pred[j].permute(1, 2, 0)).cpu())
                for row in axs: row[j].axis('off')
            plt.tight_layout()
            plt.savefig(f"/ix1/qgu/ngl18/VirtualStaining/outputs/cascade2/{epoch+1}.png")


        torch.save({
            "cascade": cascade_net.state_dict(),
            "discriminator": D2.state_dict(),
            "optimizer_G": optimizer_G.state_dict(),
            "optimizer_D": optimizer_D.state_dict(),
            "scheduler_G": scheduler_G.state_dict(),
        }, f"/ix1/qgu/ngl18/VirtualStaining/ckpts/cascade2/epoch_{epoch+1}.pth")
       
           
    scheduler_G.step()
    # scheduler_D.step()


# In[ ]:





# In[ ]:




