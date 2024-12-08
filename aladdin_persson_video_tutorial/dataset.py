# import torch
from PIL import Image
# from torchvision import transforms
import os
from torch.utils.data import Dataset
# import config 
import numpy as np


class MonetPhotoDataset(Dataset):
    
    def __init__(self, root_monet, root_photo, transform=None):
        super().__init__()

        # Path to directories
        self.root_monet = root_monet
        self.root_photo = root_photo
        self.transform = transform

        # List images in the directories
        self.monet_images = os.listdir(root_monet)
        self.photo_images = os.listdir(root_photo)

        # Check length of dataset => imp because the two datasets are not of the same size!!!
        self.length_dataset = max(len(self.monet_images), len(self.photo_images))

        self.monet_len = len(self.monet_images)
        self.photo_len = len(self.photo_images)

    
    def __len__(self):
        return self.length_dataset  # return max length

    def __getitem__(self, index):

        # Index can be greater than the dataset => modulus ensures correct range
        ## Potential flaw of the modulus - some examples might be shown more often than others
        monet_img = self.monet_images[index % self.monet_len]   # just the jpeg file
        photo_img = self.photo_images[index % self.photo_len]

        # To load jpeg file, define img path
        monet_path = os.path.join(self.root_monet, monet_img)
        photo_path = os.path.join(self.root_photo, photo_img)

        # Convert images to PIL images or numpy arrays (if using albumentations)
        monet_img = np.array(Image.open(monet_path).convert('RGB')) # convert to RGB is a safety measure in case some images are grayscale
        photo_img = np.array(Image.open(photo_path).convert('RGB'))

        if self.transform:

            # Ensure that we are performing the same transformations for monet and photo
            ## image0 because it is defined as so in config.py
            augmentations = self.transform(image=monet_img, image0=photo_img)

            monet_img = augmentations['image']
            photo_img = augmentations['image0']
        
        return monet_img, photo_img
    



