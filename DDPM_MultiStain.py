#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import torch
from torch.utils.data import Dataset
import numpy as np
import random

class MultiStainVirtualStainingDataset(Dataset):
    def __init__(self, input_images, stain_ids_dict, stain_label_dict, num_stains, transform=None):
        """
        Args:
            input_images (list): Paths to H&E tiles.
            stain_ids_dict (dict): Maps case number to list of stain label ids (e.g., [0, 1, 3]).
            stain_label_dict (dict): Maps stain name (e.g., "HER2") to label id.
            num_stains (int): Total number of stain types.
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
        stain_label_ids = self.stain_ids_dict[case_num]
        stain_idx = np.random.choice(stain_label_ids, size=1)[0]  # integer id (0, 1, 2, 3, etc.)

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
        
        return {
            "he": input_image,            # HE image (input)
            "ihc": target_image,          # Target IHC image (ground truth)
            "class_label": torch.tensor(stain_idx, dtype=torch.long)  # integer class label
        }


# In[2]:


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



stain_labels = {
    "er": 0, 
    "her2": 1,
    "ki67": 2,
    "pgr": 3
}


# In[19]:


import os, glob
root_dir = "/ix1/qgu/ngl18/ACROBAT_VirtualStaining_Valis_10x"
images = os.listdir(root_dir)[:15]
print(images)
def build_dataset(root_dir, ids, stain_labels, real_folder="he"):
    real = []
    targets = dict()
    for i in ids:
        input_images = sorted(list(glob.iglob(f"{root_dir}/{i}/{real_folder}/*.npy")))
        target_stains = list(os.listdir(f"{root_dir}/{i}/"))
        
        real.extend(input_images)
        targets[i] = [stain_labels[target] for target in target_stains if target != real_folder]

    return real, targets


# In[20]:


train_dataset_real, train_dataset_target = build_dataset(root_dir, images, stain_labels)
dataset = MultiStainVirtualStainingDataset(train_dataset_real, train_dataset_target, stain_labels, 4, transform=joint_transform)
train_dataloader = torch.utils.data.DataLoader(dataset, num_workers=16, batch_size=16, shuffle=True)


# In[21]:


from dataclasses import dataclass

@dataclass
class TrainingConfig:
    image_size = 256  # the generated image resolution
    train_batch_size = 16
    eval_batch_size = 16  # how many images to sample during evaluation
    num_epochs = 150
    gradient_accumulation_steps = 1
    learning_rate = 1e-5
    lr_warmup_steps = int(0.15 * len(train_dataloader) * num_epochs)
    save_image_epochs = 5
    save_model_epochs = 5
    num_inference_steps=1000
    mixed_precision = "fp16"  # `no` for float32, `fp16` for automatic mixed precision
    output_dir = "/ix1/qgu/ngl18/VirtualStaining/ckpts/diffusion2/"  # the model name locally and on the HF Hub
    seed = 0
config = TrainingConfig()


# In[22]:


config.lr_warmup_steps


# In[23]:


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
class_embedder = ClassEmbedder(num_classes=num_classes, embedding_dim=embedding_dim)


# In[24]:


from diffusers import UNet2DModel, DDPMScheduler, DDIMScheduler
noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000,
    beta_start=0.0001,        # <-- Force decent noise
    beta_end=0.02,            # <-- Enough noise magnitude
    beta_schedule="linear",
    variance_type="fixed_small_log"
)


# In[25]:


from diffusers import UNet2DConditionModel

model = UNet2DConditionModel(
    sample_size=config.image_size,  # the target image resolution
    in_channels=6,  # the number of input channels, 3 for RGB images
    out_channels=3,  # the number of output channels
    layers_per_block=2,  # how many ResNet layers to use per UNet block
    block_out_channels=(128, 128, 256, 256, 512, 512),  # the number of output channels for each UNet block
    down_block_types=(
        "DownBlock2D",  # a regular ResNet downsampling block
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",  # a ResNet downsampling block with spatial self-attention
        "DownBlock2D",
    ),
    up_block_types=(
        "UpBlock2D",  # a regular ResNet upsampling block
        "AttnUpBlock2D",  # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    ),
    cross_attention_dim=128
)


# In[26]:


from diffusers.optimization import get_cosine_schedule_with_warmup

optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)


# In[27]:


from diffusers import DDPMPipeline
from diffusers.utils import make_image_grid
import torchvision.transforms
import torchvision.transforms.functional as TF
import math
import os


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


import torch

def evaluate(config, epoch, pipeline, eval_dataloader, device="cuda"):
    pipeline.unet.eval()

    batch = next(iter(eval_dataloader))
    he_images = batch["he"].to(device)
    class_labels = batch["class_label"].long().to(device)
    
    with torch.no_grad():
        generated_ihc = pipeline.forward(
            he_images=he_images,
            class_labels=class_labels,
            num_inference_steps=config.num_inference_steps,
        )
    
    generated_ihc = (generated_ihc + 1) / 2
    generated_ihc = generated_ihc.clamp(0, 1)

     # --- Here is the key fix ---
    images = []
    for img in generated_ihc:
        img = TF.to_pil_image(img.cpu())
        images.append(img)

    # Make grid
    print(len(images))  # should print 16 now
    image_grid = make_image_grid(images, rows=4, cols=4)

    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"/ix1/qgu/ngl18/VirtualStaining/outputs/diffusion2/{epoch:04d}.png")


# In[28]:


from diffusers import DDPMPipeline

class HEClassDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler, class_embedder):
        super().__init__(unet=unet, scheduler=scheduler)
        self.class_embedder = class_embedder

    @torch.no_grad()
    def forward(self, he_images, class_labels, num_inference_steps=100):
        device = he_images.device
        batch_size = he_images.shape[0]
        print(he_images.shape)
        # 1. Prepare conditioning: class embeddings
        # 1. Prepare conditioning: project class labels directly
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


# In[31]:


def train_loop(config, model, class_embedder, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    
    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, class_embedder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, class_embedder, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            he_images = batch["he"]
            ihc_images = batch["ihc"]
            class_labels = batch["class_label"].long()
            
            # Sample noise to add to the images
            noise = torch.randn(he_images.shape).to(he_images.device)
            bs = he_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=he_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(ihc_images, noise, timesteps)

            # Structural conditioning: concatenate HE
            noisy_images = torch.cat([he_images, noisy_images], dim=1)  # (B, 6, 256, 256)
            class_embeddings = class_embedder(class_labels).unsqueeze(1)


            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=class_embeddings, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = HEClassDDPMPipeline(accelerator.unwrap_model(model), noise_scheduler, class_embedder)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline, train_dataloader)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                pipeline.save_pretrained(config.output_dir)
                torch.save(class_embedder.state_dict(), f"{config.output_dir}/class_embedder.pt")


# In[32]:


from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import os

from accelerate import notebook_launcher

args = (config, model, class_embedder, noise_scheduler, optimizer, train_dataloader, lr_scheduler)

notebook_launcher(train_loop, args, num_processes=1)


# In[33]:





# In[ ]:




