from typing import List, Tuple

import torchvision.transforms as transforms
from datasets.utils.logging import disable_progress_bar
from torch.utils.data import DataLoader

from flwr.common import Metrics
from flwr_datasets import FederatedDataset


disable_progress_bar()


def load_datasets(dataset_name: str, num_clients: int, batch_size: int) -> Tuple[List[DataLoader], List[DataLoader], DataLoader]:
    '''
        Loads dataset into splited train set, val set and test set.

        dataset_name: hugging face dataset name or path to local dataset
    '''
    fds = FederatedDataset(dataset=dataset_name, partitioners={
                           "train": num_clients})

    def apply_transforms(batch):
        # Instead of passing transforms to CIFAR10(..., transform=transform)
        # we will use this function to dataset.with_transform(apply_transforms)
        # The transforms object is exactly the same
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        batch["img"] = [transform(img) for img in batch["img"]]
        return batch

    # Create train/val for each partition and wrap it into DataLoader
    trainloaders = []
    valloaders = []
    for partition_id in range(num_clients):
        partition = fds.load_partition(partition_id, "train")
        partition = partition.with_transform(apply_transforms)
        partition = partition.train_test_split(train_size=0.8, seed=42)
        trainloaders.append(DataLoader(
            partition["train"], batch_size=batch_size))
        valloaders.append(DataLoader(partition["test"], batch_size=batch_size))
    testset = fds.load_split("test").with_transform(apply_transforms)
    testloader = DataLoader(testset, batch_size=batch_size)
    return trainloaders, valloaders, testloader


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}
