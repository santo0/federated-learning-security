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

DEVICE = torch.device("cpu")
MALICIOUS_CLIENTS = 1
MALICIOUS_CLIENTS_IDS = range(0, MALICIOUS_CLIENTS)
POISON_RATE = 1
TOTAL_ROUNDS = 30


def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid, net, trainloader, valloader, device, malicious_clients_ids):
        self.cid = cid
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.DEVICE = device
        self.malicious_clients_ids = malicious_clients_ids

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        nk_client_data_points = train(self.net, self.trainloader,
                                      epochs=1, DEVICE=self.DEVICE)  # this is only for elaborating the nk data points in paper, but can be done with valloader as well (orignal flwr tutorial)

        return get_parameters(self.net), nk_client_data_points, {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader, DEVICE=self.DEVICE)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}


class SimpleCNN(nn.Module):
    def __init__(self) -> None:
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 13 * 13)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def train(net, trainloader, epochs: int, DEVICE):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    nk_data_points = 0
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= total
        epoch_acc = correct / total
        nk_data_points = total

    return nk_data_points


def test(net, testloader, DEVICE):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def test_server(net, testloader, DEVICE):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    attack_success, correct_backdoor = 0, 0
    net.eval()
    has_shown = False
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # try backdoor attack
            pattern = getGlobalTrigger(images[0][0].squeeze().shape)
            for image in images:
                # plt.imshow(image.T.squeeze(), cmap='gray')
                # plt.show()
                image += pattern
                # plt.imshow(image.T.squeeze(), cmap='gray')
                # plt.show()
                # raise Exception("stop")
            if has_shown == False:
                has_shown = True
                # plt.imshow(images[-1].squeeze().T, cmap='gray')
                # plt.show()
            outputs = net(images)
            # print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            # check that predicted is the backdoor class
            # print(predicted)
            correct_backdoor += (predicted == 0).sum().item()

    loss /= len(testloader.dataset)
    accuracy = correct / total
    attack_success = correct_backdoor / total
    return loss, accuracy, attack_success


class AttackBackdoor(torch.utils.data.Dataset):
    def __init__(self, dataset, class_ids_to_poison, attack_pattern, backdoor_target_class_id):
        self.dataset = dataset
        self.class_ids = class_ids_to_poison
        self.attack_pattern = attack_pattern
        self.target_class_id = backdoor_target_class_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if y in self.class_ids:
            # if poison_rate is 0.3, then 30% of the images will be poisoned
            if np.random.rand() < POISON_RATE:
                y = self.target_class_id
                x += self.attack_pattern
        return x, y


def getGlobalTrigger(shape):
    # print(f"shape is {shape}")
    pattern = torch.zeros(shape)
    # pattern[22:,22:] = 255
    # square
    pattern[4:8, 4:8] = 255
    pattern[10:14, 4:8] = 255
    pattern[4:8, 10:14] = 255
    pattern[10:14, 10:14] = 255
    return pattern


def getLocalTrigger(shape):
    # print(f"shape is {shape}")
    pattern_1 = torch.zeros(shape)
    pattern_1[4:8, 4:8] = 255
    pattern_2 = torch.zeros(shape)
    pattern_2[10:14, 4:8] = 255
    pattern_3 = torch.zeros(shape)
    pattern_3[4:8, 10:14] = 255
    pattern_4 = torch.zeros(shape)
    pattern_4[10:14, 10:14] = 255
    return pattern_1, pattern_2, pattern_3, pattern_4


def iid_split(num_clients: int, storage_dir: str, batch_size=32):

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=storage_dir, transform=transform)

    # split dataset into train and test
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    trainset, testset = random_split(dataset, [train_size, test_size])

    # Split training set into `num_clients` partitions to simulate different local datasets
    partition_size = len(trainset) // num_clients
    lengths = [partition_size] * num_clients
    # add rest in first partition
    lengths[0] = lengths[0] + (len(trainset) % lengths[0])
    partitions_ds = random_split(
        trainset, lengths, torch.Generator().manual_seed(42))

    # Split each partition into train/val and create DataLoader
    train_datasets = []
    val_datasets = []
    for ds in partitions_ds:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(
            ds, lengths, torch.Generator().manual_seed(42))
        train_datasets.append(ds_train)
        val_datasets.append(ds_val)

        # trainloaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        # valloaders.append(DataLoader(ds_val, batch_size=batch_size))
    # tloader = DataLoader(testset, batch_size=batch_size)
    return train_datasets, val_datasets, testset


if __name__ == '__main__':
    # Create a dataset from the og_vehicles folder (it has images of vehicles)

    train_dsets, val_dset, server_test_dset = iid_split(10, 'og_vehicles/')
    # print(train_dsets, val_dset, server_test_dset)
    # from first subset, first image, get shape (3 channels RGB)
    for cid in range(len(train_dsets)):
        if cid in MALICIOUS_CLIENTS_IDS:
            print(f"Injecting backdoor to client {cid}")
            s = train_dsets[cid][0][0][0].squeeze().shape
            backdor_dataset = AttackBackdoor(
                dataset=train_dsets[cid], class_ids_to_poison=[0, 1],
                attack_pattern=getGlobalTrigger(s), backdoor_target_class_id=0)
            train_dsets[cid] = backdor_dataset
            # for i in range(len(train_dsets[cid])):
            #     print(f"cid {cid}  class {train_dsets[cid][i][1]}")
    # visualize random image of backdor_dataset
            # plt.imshow(backdor_dataset[0][0].T.squeeze(), cmap='gray')
            # plt.show()
    trainloaders = []
    valloaders = []
    for cid in range(len(train_dsets)):
        trainloaders.append(DataLoader(
            train_dsets[cid], batch_size=32, shuffle=True))
        valloaders.append(DataLoader(val_dset[cid], batch_size=32))
    testloader = DataLoader(server_test_dset, batch_size=32)

    def getClient(cid) -> FlowerClient:
        net = SimpleCNN().to(DEVICE)
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        return FlowerClient(cid, net, trainloader, valloader, DEVICE, malicious_clients_ids=MALICIOUS_CLIENTS_IDS)

    # The `evaluate` function will be by Flower called after every round
    def evaluateGlobalModel(server_round, parameters, config):
        net = SimpleCNN()
        valloader = testloader
        # Update model with the latest parameters
        set_parameters(net, parameters)
        loss, accuracy, attack_success = test_server(
            net, valloader, DEVICE=DEVICE)
        print(
            f"Server-side evaluation loss {loss} / accuracy {accuracy} / attack success {attack_success}")
        return loss, {"accuracy": accuracy, "attack_success": attack_success}

    fl.simulation.start_simulation(
        client_fn=getClient,
        num_clients=10,
        config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=10,
            min_evaluate_clients=10,
            min_available_clients=10,
            evaluate_metrics_aggregation_fn=weighted_average,
            evaluate_fn=evaluateGlobalModel,
        ),
    )
