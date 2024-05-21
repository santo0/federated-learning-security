from utils import weighted_average

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torch
from torchvision import transforms

import torch
from torchvision import datasets, transforms

import flwr as fl
from clients import DEVICE


import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import List
from collections import OrderedDict
import flwr as fl
import numpy as np
import torch
from model import SimpleCNN, train, test


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, device, malicious_clients_ids, epochs):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.DEVICE = device
        self.malicious_clients_ids = malicious_clients_ids
        self.epochs = epochs

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        nk_client_data_points = train(self.net, self.trainloader,
                                      epochs=self.epochs, DEVICE=self.DEVICE) 

        return get_parameters(self.net), nk_client_data_points, {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, DEVICE=self.DEVICE)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
