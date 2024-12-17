# 02456 Deep Learning Final Project
## Mimicking art style using CycleGAN

### Group members
- Benjamin Pedersen Rasmussen (s234911)
- Raquel Moleiro Marques (s243636)
- Sree Keerthi Desu (s243933)
- Ting-Hui Cheng (s232855)

## Project Overview
This project uses CycleGANs to perform artistic style transfer, transforming real-world photographs into Monet-style paintings without the need for paired datasets
We tackle key challenges like checkerboard artifacts and limitations in traditional GAN structures to ensure high-quality results.

## Code Structure
- `main.ipynb`: Dataset overview, training and testing the model.
- `utils/Dataloader.py`: Loads and preprocesses the dataset.
- `utils/Architecture`: Contains the generator and discriminator architectures.
- `utils/Model.py`: Implements the CycleGAN model.
- `utils/FixedConv.py`, `utils/WeightNormalization`: Contains classes for model improvements.