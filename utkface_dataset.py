# +
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.gridspec import GridSpec
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
import torch.optim as optim
import torch.utils.data as data, torchvision as tv
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from torchvision.io import read_image
from torchvision import models
from torchvision.models import resnet50, resnet152, resnet18, ResNet50_Weights
from torchvision.datasets import ImageFolder
from torchmetrics.classification import Accuracy
from torchvision.transforms import ToTensor, Lambda
from pytorch_lightning.callbacks import ModelCheckpoint

from torch.utils.data import (
    TensorDataset,
    DataLoader,
    Dataset,
    random_split,
    WeightedRandomSampler,
)
from torchsummary import summary

import pytorch_lightning as pl

import os
import re
import random
import pickle
import glob
import shutil

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

import time
from collections import Counter
from scipy import stats

from lime import lime_image

import requests
from PIL import Image
from PIL import ImageFile
from io import BytesIO

from skimage import io, transform
from skimage.transform import rotate
from skimage.util import random_noise
from skimage.filters import gaussian
from skimage.segmentation import mark_boundaries

import warnings

warnings.filterwarnings("ignore")

# -

class UTKFaceDataset(pl.LightningDataModule):
    """
    Class for loading the UTKFace dataset
    """

    def __init__(self, img_dir: str, batch_size: int = 32, image_size: tuple = (224, 224)):
        """
        Constructor for UTKFaceDataset class

        Parameters
        ----------
        img_dir : str
            Directory where the dataset is located
        batch_size : int, optional
            Batch size for the dataloaders, by default 32
        image_size : tuple, optional
            Size of the images, by default (224, 224)
        """

        super().__init__()
        self.img_dir = img_dir
        self.batch_size = batch_size
        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def show_random_images(self, num_images: int = 4) -> None:
        """
        Shows a random sample of images from the dataset

        Parameters
        ----------
        num_images : int, optional
            Number of images to show, by default 4
        """

        unique_classes = set()
        while len(unique_classes) < num_images:
            _, label = self.dataset[random.randint(0, len(self.dataset) - 1)]
            unique_classes.add(label)

        images = []
        labels = []
        for class_idx in unique_classes:
            class_images = [
                i for i, j in enumerate(self.dataset.targets) if j == class_idx
            ]
            image_idx = random.choice(class_images)
            image, label = self.dataset[image_idx]
            images.append(image)
            labels.append(label)

        images_tensor = torch.stack(images)

        self._imshow(
            torchvision.utils.make_grid(images_tensor),
            [self.dataset.classes[lbl] for lbl in labels],
        )

    def _imshow(self, img: torch.Tensor, titles: list) -> None:
        """
        Shows a matplotlib figure with the specified images

        Parameters
        ----------
        img : torch.Tensor
            Images to show
        titles : list
            Titles for the images
        """

        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)

        plt.figure(figsize=(12, 6))
        plt.imshow(img)
        plt.title(" - ".join(titles))
        plt.show()

    def setup(self, stage: str = None) -> None:
        """
        Loads the dataset

        Parameters
        ----------
        stage : str, optional
            Stage of the training, by default None
        """

        self.dataset = ImageFolder(root=self.img_dir, transform=self.transform)

    def __len__(self) -> int:
        """
        Returns the length of the dataset

        Returns
        -------
        int
            Length of the dataset
        """

        return len(self.dataset)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns the image and label at the specified index

        Parameters
        ----------
        idx : int
            Index of the image to retrieve

        Returns
        -------
        tuple
            Image and label at the specified index
        """

        return self.dataset[idx]

    def train_dataloader(self) -> DataLoader:
        """
        Returns a dataloader for the training set

        Returns
        -------
        DataLoader
            Dataloader for the training set
        """

        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self) -> DataLoader:
        """
        Returns a dataloader for the validation set

        Returns
        -------
        DataLoader
            Dataloader for the validation set
        """

        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def test_dataloader(self) -> DataLoader:
        """
        Returns a dataloader for the test set

        Returns
        -------
        DataLoader
            Dataloader for the test set
        """

        return DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


class UTKFaceMergedDataset(Dataset):
    """Dataset class for the UTKFaceMerged dataset.

    Attributes:
        dataframe (pandas.DataFrame): The dataframe containing the dataset information.
        img_dir (str): The directory where the images are stored.
        image_size (tuple): The desired size of the images.
        transform (torchvision.transforms.Compose): A series of transformations to be applied to the images.

    """

    def __init__(self, dataframe: pd.DataFrame, img_dir: str, image_size: tuple = (224, 224)):
        """Initializes the dataset.

        Args:
            dataframe (pandas.DataFrame): The dataframe containing the dataset information.
            img_dir (str): The directory where the images are stored.
            image_size (tuple, optional): The desired size of the images. Defaults to (224, 224).
        """
        self.dataframe = dataframe
        self.img_dir = img_dir
        self.image_size = image_size
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.dataframe)

    def __getitem__(self, idx):
        """Gets the item at the given index.

        Args:
            idx (int): The index of the item to get.

        Returns:
            tuple: A tuple containing the image, gender, and age.
        """
        img_name = self.dataframe.iloc[idx]["full_path"]
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)

        image = transforms.ToPILImage()(image)
        image = transforms.Resize(self.image_size)(image)

        if image.mode == "L":
            image = transforms.Grayscale(num_output_channels=3)(image)

        if image.mode == "RGBA":
            image = image.convert("RGB")

        image = transforms.ToTensor()(image)
        image = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )(image)

        gender = self.dataframe.iloc[idx]["gender"]
        age = self.dataframe.iloc[idx]["age"]

        gender = 1.0 if gender == "female" else 0.0

        gender = torch.tensor(gender, dtype=torch.float32)
        age = torch.tensor(age, dtype=torch.float32)

        return image, gender, age


