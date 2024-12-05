# import torch
from PIL import Image
# from torchvision import transforms
import os
from torch.utils.data import Dataset
# import config 
import numpy as np


class HorseZebraDataset(Dataset):
    
    def __init__(self, root_zebra, root_horse, transform=None):
        super().__init__()

        # Path to directories
        self.root_zebra = root_zebra
        self.root_horse = root_horse
        self.transform = transform

        # List images in the directories
        self.zebra_images = os.listdir(root_zebra)
        self.horse_images = os.listdir(root_horse)

        # Check length of dataset => imp because the two datasets are not of the same size!!!
        self.length_dataset = max(len(self.zebra_images), len(self.horse_images))

        self.zebra_len = len(self.zebra_images)
        self.horse_len = len(self.horse_images)

    
    def __len__(self):
        return self.length_dataset  # return max length

    def __getitem__(self, index):

        # Index can be greater than the dataset => modulus ensures correct range
        ## Potential flaw of the modulus - some examples might be shown more often than others
        zebra_img = self.zebra_images[index % self.zebra_len]   # just the jpeg file
        horse_img = self.horse_images[index % self.horse_len]

        # To load jpeg file, define img path
        zebra_path = os.path.join(self.root_zebra, zebra_img)
        horse_path = os.path.join(self.root_horse, horse_img)

        # Convert images to PIL images or numpy arrays (if using albumentations)
        zebra_img = np.array(Image.open(zebra_path).convert('RGB')) # convert to RGB is a safety measure in case some images are grayscale
        horse_img = np.array(Image.open(horse_path).convert('RGB'))

        if self.transform:

            # Ensure that we are performing the same transformations for zebra and horse
            ## image0 because it is defined as so in config.py
            aumentations = self.transform(image=zebra_img, image0=horse_img)

            zebra_img = aumentations['image']
            horse_img = aumentations['image0']
        
        return zebra_img, horse_img
    



