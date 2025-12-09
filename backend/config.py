import torch

MODEL_PATH = "D:/Label_studio/resunet_a_checkpoint_epoch_500.pth"
IMG_SIZE = 256

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
