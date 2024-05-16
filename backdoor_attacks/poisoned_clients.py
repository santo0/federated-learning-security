from collections import OrderedDict
from typing import List

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from clients import FlowerClient, DEVICE, Net

def get_client_fn(
        train_loaders: List[DataLoader], val_loaders: List[DataLoader]) -> FlowerClient:
    # loaders injection
    def client_fn(cid: str) -> FlowerClient:
        """Create a Flower client representing a single organization."""

        # Load model
        net = Net().to(DEVICE)

        # Load data (CIFAR-10)
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        train_loader = train_loaders[int(cid)]
        val_loader = val_loaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(net, train_loader, val_loader).to_client()
    return client_fn
