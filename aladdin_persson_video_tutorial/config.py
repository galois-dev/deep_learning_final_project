import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 1  # taken from paper
LEARNING_RATE = 1e-5    # taken from paper
LAMBDA_IDENTITY = 0.0   # taken from paper
LAMBDA_CYCLE = 10   # taken from paper
NUM_WORKERS = 4 # taken from paper
NUM_EPOCHS = 10 # taken from paper
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_PHOTO = "gen_photo.pth.tar"
CHECKPOINT_GEN_MONET = "gen_monet.pth.tar"
CHECKPOINT_CRITIC_PHOTO = "critic_photo.pth.tar"
CHECKPOINT_CRITIC_MONET = "critic_monet.pth.tar"

transforms = A.Compose(
    [
        A.Resize(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.ColorJitter(p=0.1),   # Optional!! was NOT initially there!
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
        ToTensorV2(),
    ],
    additional_targets={"image0": "image"},
)
