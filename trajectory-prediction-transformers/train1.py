import torch
import torch.nn.functional as F
from tqdm import tqdm 
import numpy as np
import os
import dataloader 
import model
import utils
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # defining model save location
    save_location = "./models"

    # defining dataset locations
    dataset_folder = "./datasets"
    dataset_name = "raw"

    # setting validation size. if val_size = 0, split percentage is 80-20
    val_size = 0
    # length of sequence given to encoder
    gt = 8
    # length of sequence given to decoder
    horizon = 12

    