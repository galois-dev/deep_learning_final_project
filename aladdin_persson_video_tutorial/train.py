import torch
from dataset import MonetPhotoDataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import os


def train_fn(disc_photo, disc_monet, gen_monet, gen_photo, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler):
    
    # To get a progress bar
    loop = tqdm(loader, leave=True)

    # Training
    for idx, (monet, photo) in enumerate(loop): # since the tuple is the way it is returned in the dataset 
        monet = monet.to(config.DEVICE)
        photo = photo.to(config.DEVICE)

        # Train discriminators H and Z
        with torch.cuda.amp.autocast(): # necessary for Float16!!!

            # Generate a fake photo
            fake_photo = gen_photo(monet)
            D_photo_real = disc_photo(photo)
            D_photo_fake = disc_photo(fake_photo.detach())  # detach() ensure that we dont have to generate another fake_photo later when training generator
            D_photo_real_loss = mse(D_photo_real, torch.ones_like(D_photo_real)) # real = 1, fake = 0
            D_photo_fake_loss = mse(D_photo_fake, torch.zeros_like(D_photo_fake)) # real = 1, fake = 0
            D_photo_loss = D_photo_real_loss + D_photo_fake_loss    # actual loss

            # Generate a fake monet
            fake_monet = gen_monet(photo)
            D_monet_real = disc_monet(monet)
            D_monet_fake = disc_monet(fake_monet.detach())  # detach() ensure that we dont have to generate another fake_monet later when training generator
            D_monet_real_loss = mse(D_monet_real, torch.ones_like(D_monet_real)) # real = 1, fake = 0
            D_monet_fake_loss = mse(D_monet_fake, torch.zeros_like(D_monet_fake)) # real = 1, fake = 0
            D_monet_loss = D_monet_real_loss + D_monet_fake_loss    # actual loss

            # Total loss
            D_loss = (D_photo_loss + D_monet_loss)/2    # not sure why dividing by 2
        
        # Update discriminators in the right direction
        opt_disc.zero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()


        # Train generators H and Z
        with torch.cuda.amp.autocast():
            D_photo_fake = disc_photo(fake_photo)
            D_monet_fake = disc_monet(fake_monet)

            # To "fool" the discriminator... 
            ## Adversarial loss for both generators
            loss_G_photo = mse(D_photo_fake, torch.ones_like(D_photo_fake))
            loss_G_monet = mse(D_monet_fake, torch.ones_like(D_monet_fake)) # fake = 0, real = 1 => only ones_like since we want to fool the discriminator into thinking it's real!

            ## Cycle loss for both generators
            cycle_monet = gen_monet(fake_photo)  # trying to generate a monet using a fake photo => should HOPEFULLY give back og monet
            cycle_photo = gen_photo(fake_monet) # similarly, should HOPEFULLY give back og photo from generated monet
            cycle_monet_loss = l1(monet, cycle_monet)
            cycle_photo_loss = l1(photo, cycle_photo)

            ## Idenitity loss for both generators
            # identity_monet = gen_monet(monet)   # this is if we send in a monet to the one that should already generate a monet
            # identity_photo = gen_photo(photo)
            # identity_monet_loss = l1(monet, identity_monet) # CAN BE REMOVED! unnecessary since LAMDA_IDENTITY = 0 in config
            # identity_photo_loss = l1(photo, identity_photo) # CAN BE REMOVED! unnecessary since LAMDA_IDENTITY = 0 in config

            ## Add all losses
            G_loss = (
                loss_G_monet
                + loss_G_photo
                + cycle_monet_loss * config.LAMBDA_CYCLE
                + cycle_photo_loss * config.LAMBDA_CYCLE
                # + identity_photo_loss * config.LAMBDA_IDENTITY
                # + identity_monet_loss * config.LAMBDA_IDENTITY
            )
        
        # Update generators in the right direction
        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        # Show image
        if idx % 200 == 0:
            save_image(fake_photo*0.5 + 0.5, f'saved_images/photo_{idx}.png')    # the 0.5 is to emulate the inverse of the normalization to get the correct coloring
            save_image(fake_monet*0.5 + 0.5, f'saved_images/monet_{idx}.png')


def main():

    # Initialize discriminator
    disc_photo = Discriminator(in_channels=3).to(config.DEVICE)
    disc_monet = Discriminator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_photo.parameters()) + list(disc_monet.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    # Initialize generator
    gen_monet = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)    # H -> Z
    gen_photo = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)    # Z -> H
    opt_gen = optim.Adam(
        list(gen_monet.parameters()) + list(gen_photo.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    # Use L1 loss for cycle consistency loss and identity loss
    L1 = nn.L1Loss()

    # Use MSE for residual loss
    mse = nn.MSELoss()

    # Check if we should load a checkpoint, then load both generators and discriminators (critic)
    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_PHOTO,
            gen_photo,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_MONET,
            gen_monet,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_PHOTO,
            disc_photo,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_MONET,
            disc_monet,
            opt_disc,
            config.LEARNING_RATE,
        )

    # Create datasets
    dataset = MonetPhotoDataset(
        root_photo=config.TRAIN_DIR + "/photo_jpg",
        root_monet=config.TRAIN_DIR + "/monet_jpg",
        transform=config.transforms,
    )
    # ## To evaluate model, load val
    # val_dataset = MonetPhotoDataset(
    #     root_photo="cyclegan_test/photo1",
    #     root_monet="cyclegan_test/monet1",
    #     transform=config.transforms,
    # )

    # # Load datasets
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     pin_memory=True,
    # )
    
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # For Float16 training => can be be removed to run in Float32!
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # Train
    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_photo,
            disc_monet,
            gen_monet,
            gen_photo,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        # At each epoch, save generators and discriminators under a checkpoint
        if config.SAVE_MODEL:
            save_checkpoint(gen_photo, opt_gen, filename=config.CHECKPOINT_GEN_PHOTO)
            save_checkpoint(gen_monet, opt_gen, filename=config.CHECKPOINT_GEN_MONET)
            save_checkpoint(disc_photo, opt_disc, filename=config.CHECKPOINT_CRITIC_PHOTO)
            save_checkpoint(disc_monet, opt_disc, filename=config.CHECKPOINT_CRITIC_MONET)

if __name__=='__main__':
    main()