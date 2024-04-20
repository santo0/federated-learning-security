from collections import OrderedDict
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics
from flwr_datasets import FederatedDataset

from utils import DEVICE, get_client_fn, NUM_CLIENTS, weighted_average, load_datasets


if __name__ == '__main__':
    train_loaders, val_loaders, test_loader = load_datasets()
    # Create FedAvg strategy
    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=10,
        min_evaluate_clients=5,
        min_available_clients=10,
        # <-- pass the metric aggregation function
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # Specify the resources each of your clients need. By default, each
    # client will be allocated 1x CPU and 0x GPUs
    client_resources = {"num_cpus": 1, "num_gpus": 0.0}
    if DEVICE.type == "cuda":
        # here we are assigning an entire GPU for each client.
        client_resources = {"num_cpus": 1, "num_gpus": 1.0}
        # Refer to our documentation for more details about Flower Simulations
        # and how to setup these `client_resources`.

    # Start simulation
    fl.simulation.start_simulation(
        # TODO: For malicious clients: make factory of client given NUM_CLIENTS and BAD ratio (3 out of 10  will be attackers)
        # For attacking clients, the FlowerClient will be different
        client_fn=get_client_fn(train_loaders, val_loaders),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources=client_resources,
    )
