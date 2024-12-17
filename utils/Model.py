import torch
from torch import nn
from itertools import chain
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from utils.Architecture import Generator, Discriminator

class Model(torch.nn.Module):
    def __init__(self, epochs, device, cycle_lambda = 10, Learning_rate=2e-4):
        super(Model,self).__init__()
        self.device = device
        
        self.gen_monet2photo = Generator(input_channel=3).to(self.device)
        self.gen_photo2monet = Generator(input_channel=3).to(self.device)
        self.disc_monet = Discriminator(input_channel=3).to(self.device)
        self.disc_photo = Discriminator(input_channel=3).to(self.device)

        # Optimizer for the generator and discriminators
        self.opt_gen = torch.optim.Adam(chain(self.gen_monet2photo.parameters(), self.gen_photo2monet.parameters()),
                                         lr = Learning_rate,
                                         betas= (0.5, 0.999))
        self.opt_disc = torch.optim.Adam(chain(self.disc_monet.parameters(), self.disc_photo.parameters()),
                                         lr = Learning_rate,
                                         betas= (0.5, 0.999))
        
        self.L1 = nn.L1Loss().to(self.device)
        self.mse = nn.MSELoss().to(self.device) 
        self.cycle_lambda = cycle_lambda
        self.epochs = epochs

        self.D_scaler = GradScaler()
        self.G_scaler = GradScaler()
        
        self.zero_tensor = None
        self.Tensor = torch.FloatTensor
    def init_weights(self):
        pass

    # Generate and add noise to the inputs of the discriminator
    def add_noise(self, tensor, noise_level=0.05):
        """
        Adds Gaussian noise to the input tensor.
        Args:
            tensor (torch.Tensor): Input tensor to which noise will be added.
            noise_level (float): Standard deviation of the Gaussian noise.
        Returns:
            torch.Tensor: Tensor with added noise.
        """
        if noise_level > 0:
            noise = torch.randn_like(tensor) * noise_level
            return tensor + noise
        return tensor
    
    """ From https://github.com/NVlabs/SPADE/blob/master/models/networks/loss.py """
    def get_zero_tensor(self, input):
        if self.zero_tensor is None:
            self.zero_tensor = self.Tensor(1).fill_(0)
            self.zero_tensor.requires_grad_(False)
        return self.zero_tensor.expand_as(input)
    def hinge_loss(self, input, target_is_real, for_discriminator = False):
        if for_discriminator:
            if target_is_real:
                minval = torch.min(input - 1, self.get_zero_tensor(input))
                loss = -torch.mean(minval)
            else:
                minval = torch.min(-input - 1, self.get_zero_tensor(input))
                loss = -torch.mean(minval)
        else:
            assert target_is_real, "The generator's hinge loss must be aiming for real"
            loss = -torch.mean(input)
        return loss
    
    def train(self, dataloader):
        for epoch in range(self.epochs):
            avg_G_loss = 0.0
            avg_D_loss = 0.0

            reals = 0
            fakes = 0
            print("\nEpoch {}/{}".format(epoch+1, self.epochs))
            loop = tqdm(dataloader, leave=True, total=len(dataloader))
            noise_level = max(0.05, 0.05 * (1 - epoch / self.epochs))
            for i, (photo, monet) in enumerate(loop):
                photo, monet = photo.to(self.device), monet.to(self.device)
                
                if i >= len(dataloader) - 1:  # Ensure no more than the intended batches
                    break

                # Train discriminators Between photo and monet
                with autocast():
                    # Generate fake photo to train photos discriminator 
                    fake_photo = self.gen_monet2photo(monet)
                    D_photo = self.disc_photo(self.add_noise(photo, noise_level))
                    D_fake_photo = self.disc_photo(self.add_noise(fake_photo.detach(), noise_level))
                    reals += D_photo.mean().item()
                    fakes += D_fake_photo.mean().item()
                    # Use the mean squared error between the discriminator output
                    # and the label (Real photo is 1 and the generated one is 0)
                    D_real_photo_loss = self.hinge_loss(D_photo, True, True)
                    D_fake_photo_loss = self.hinge_loss(D_fake_photo, False, True)

                    D_photo_loss = (D_real_photo_loss + D_fake_photo_loss) / 2

                    # Generate fake monet pictures to train monets discriminator
                    fake_monet = self.gen_photo2monet(photo)
                    D_monet = self.disc_monet(self.add_noise(monet, noise_level))
                    D_fake_monet = self.disc_monet(self.add_noise(fake_monet.detach(), noise_level))
                    # Use the mean squared error between the discriminator output
                    # and the label (Real monet is 1 and the generated monet is 0)
                    D_real_monet_loss = self.hinge_loss(D_monet,  True, True)
                    D_fake_monet_loss = self.hinge_loss(D_fake_monet, False, True)
                    D_monet_loss = (D_real_monet_loss + D_fake_monet_loss) / 2
                    total_D_loss = D_monet_loss + D_photo_loss
                    avg_D_loss += total_D_loss.item()
                self.opt_disc.zero_grad()
                self.D_scaler.scale(total_D_loss).backward()  
                self.D_scaler.step(self.opt_disc)  
                self.D_scaler.update() 
                

                # Train generators for photo and monet images
                with autocast():

                    # adversarial losses
                    D_fake_photo = self.disc_photo(self.add_noise(fake_photo.detach(), noise_level))
                    D_fake_monet = self.disc_monet(self.add_noise(fake_monet.detach(), noise_level))
                    G_fake_photo_loss = self.hinge_loss(D_fake_photo, True, False)
                    G_fake_monet_loss = self.hinge_loss(D_fake_monet, True, False)
                    # cycle losses
                    cycle_monet = self.gen_photo2monet(fake_photo)
                    cycle_photo = self.gen_monet2photo(fake_monet)
                   
                    cycle_monet_loss = self.L1(monet, cycle_monet)
                    cycle_photo_loss = self.L1(photo, cycle_photo)
    
                    # total loss
                    total_G_loss = (
                        G_fake_photo_loss
                        + G_fake_monet_loss
                        + cycle_monet_loss * self.cycle_lambda
                        + cycle_photo_loss * self.cycle_lambda
                    )
                    avg_G_loss += total_G_loss.item()
                self.opt_gen.zero_grad()
                self.G_scaler.scale(total_G_loss).backward()  
                self.G_scaler.step(self.opt_gen)  
                self.G_scaler.update()
                loop.set_postfix(Monet_Real=reals / (i + 1), Monet_fake=fakes / (i + 1))
            print("Generator Loss:%f  -  Discriminator Loss:%f" % (avg_G_loss, avg_D_loss))
            
    def test(self, photo, monet):
        results = []
        
        photo, monet = photo.to(self.device), monet.to(self.device)

        generated_monet = self.gen_photo2monet(photo)

        # Pass the monet and generated images through the discriminator
        generated_score = self.disc_monet(generated_monet.detach())
        real_score = self.disc_monet(monet)
        generated_score = generated_score.mean().item()
        real_score = real_score.mean().item()
        
        results.append({
            "generated_score": generated_score,
            "real_score": real_score,
            "generated_image": generated_monet.detach().cpu()  # Detach and move to CPU
        })
        return results
        