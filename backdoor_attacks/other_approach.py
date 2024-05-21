from utils import weighted_average
from copy import copy
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset
import torch
from torchvision import transforms

import torch
from torchvision import datasets, transforms

import flwr as fl
from clients import DEVICE

import torch

import flwr as fl
import numpy as np
import torch
from model import SimpleCNN, train, test
from client import FlowerClient, get_parameters, set_parameters

DEVICE = torch.device("cpu")
MALICIOUS_CLIENTS = 0
MALICIOUS_CLIENTS_IDS = [x for x in range(0, MALICIOUS_CLIENTS)]
print("MALICIOUS_CLIENTS", MALICIOUS_CLIENTS_IDS)
POISON_RATE = 0.0
TOTAL_ROUNDS = 30
NUM_CLIENTS = 10
BACKDOOR_CLASS = 1

# 3,0.3


def test_server(net, testloader, poison_testloader):
    """Evaluate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    correct_backdoor, total_backdoor = 0, 0
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
        for images, labels in poison_testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_backdoor += labels.size(0)
            correct_backdoor += (predicted == BACKDOOR_CLASS).sum().item()

            # for image in images:
            # plt.imshow(image.T.squeeze(), cmap='gray')
            # plt.show()
            # image += pattern
            # plt.imshow(image.T.squeeze(), cmap='gray')
            # plt.show()
            # raise Exception("stop")

            if has_shown == False:
                has_shown = True
                # show all images in the batch in a grid
                fig, axes = plt.subplots(5, 7, figsize=(10, 10))
                axes = axes.flatten()
                for i in range(len(images)):
                    axes[i].imshow(images[i].squeeze().T, cmap='gray')
                plt.tight_layout()
                plt.show()
                print(predicted, labels)
                # plt.imshow(images[-1].squeeze().T, cmap='gray')
                # plt.show()

    loss /= len(testloader.dataset)
    accuracy = correct / total
    attack_success = correct_backdoor / total_backdoor
    return loss, accuracy, attack_success


class AttackBackdoor(torch.utils.data.Dataset):
    def __init__(self, dataset, class_ids_to_poison, attack_pattern, backdoor_target_class_id, poison_rate):
        self.dataset = copy(dataset)
        self.class_ids = class_ids_to_poison
        self.attack_pattern = attack_pattern
        self.target_class_id = backdoor_target_class_id
        self.poison_rate = poison_rate

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        if y in self.class_ids:
            # if poison_rate is 0.3, then 30% of the images will be poisoned
            if np.random.rand() < self.poison_rate:
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

    return train_datasets, val_datasets, testset


def filter_targets(dataset: Subset):
    indices = [i for i in dataset.indices
               if dataset.dataset.imgs[i][1] != BACKDOOR_CLASS]
    return Subset(dataset.dataset, indices)


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
                dataset=train_dsets[cid],
                class_ids_to_poison=[0 if BACKDOOR_CLASS == 1 else 1],
                attack_pattern=getGlobalTrigger(s), backdoor_target_class_id=BACKDOOR_CLASS, poison_rate=POISON_RATE)
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
    s = server_test_dset[0][0][0].squeeze().shape

    poison_testloader = DataLoader(AttackBackdoor(
        dataset=filter_targets(server_test_dset),
        class_ids_to_poison=[0 if BACKDOOR_CLASS == 1 else 1],
        attack_pattern=getGlobalTrigger(s), backdoor_target_class_id=BACKDOOR_CLASS, poison_rate=1),
        batch_size=32
    )

    def getClient(cid) -> FlowerClient:
        net = SimpleCNN().to(DEVICE)
        trainloader = trainloaders[int(cid)]
        valloader = valloaders[int(cid)]
        return FlowerClient(cid, net, trainloader, valloader, DEVICE,
                            malicious_clients_ids=MALICIOUS_CLIENTS_IDS, epochs=1)

    # The `evaluate` function will be by Flower called after every round
    def evaluateGlobalModel(server_round, parameters, config):
        net = SimpleCNN()
        # Update model with the latest parameters
        set_parameters(net, parameters)
        loss, accuracy, attack_success = test_server(
            net, testloader, poison_testloader)
        print(
            f"Server-side evaluation loss {loss} / accuracy {accuracy} / attack success {attack_success}")
        return loss, {"accuracy": accuracy, "attack_success": attack_success}
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=NUM_CLIENTS,
        min_evaluate_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=evaluateGlobalModel,
    )

    fl.simulation.start_simulation(
        client_fn=getClient,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=TOTAL_ROUNDS),
        strategy=strategy,
    )
