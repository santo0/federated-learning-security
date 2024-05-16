import os
import shutil
import random

from datasets import load_dataset
import datasets
from torchvision.datasets import ImageFolder
from flwr_datasets.partitioner import IidPartitioner
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

import torch
from torchvision import datasets, transforms

train_data_path = './local_vehicles/train/'
test_data_path = './local_vehicles/test/'

train_poisoned_data_path = './poisoned_vehicles/train/'
test_poisoned_data_path = './poisoned_vehicles/test/'


transform = transforms.Compose(
    [
    #transforms.Resize((32, 32)),
     transforms.ToTensor(),
     #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])


def show_random_sample(dataset):

    # Select a random image from the trainset
    random_index = random.randint(0, len(dataset)-1)
    image, label = dataset[random_index]

    # Denormalize the image tensor
    #mean = torch.tensor([0.5, 0.5, 0.5])
    #std = torch.tensor([0.5, 0.5, 0.5])
    #image = transforms.functional.normalize(image, (-mean / std), (1.0 / std))
    # Convert tensor to numpy array and transpose dimensions
    image_np = image.numpy().transpose((1, 2, 0))

    # Display the image and its corresponding label
    plt.imshow(image_np)
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()


def get_vehicle_dataset():
    trainset = DataLoader(ImageFolder(root=train_data_path, transform=transform), batch_size=32, shuffle=False)
    testset = DataLoader(ImageFolder(root=test_data_path, transform=transform), batch_size=32, shuffle=False)
    return trainset, testset


def get_vehicle_imagefolder():
    trainset = ImageFolder(root=train_data_path, transform=transform)
    testset = ImageFolder(root=test_data_path, transform=transform)
    return trainset, testset

def get_poisoned_vehicle_dataset():
    trainset = ImageFolder(root=train_poisoned_data_path, transform=transform)
    testset = ImageFolder(root=test_poisoned_data_path, transform=transform)
    return trainset, testset
#    dataset = load_dataset('train_poisoned_vehicles',
#                           data_dir='./poisoned_vehicles/train/')
#    return dataset


def split_images(input_dir, output_train_dir, output_test_dir, split_ratio=0.7):
    # Create train and test directories if they don't exist
    os.makedirs(output_train_dir, exist_ok=True)
    os.makedirs(output_test_dir, exist_ok=True)

    # Loop through each class directory in the input directory
    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if os.path.isdir(class_dir):
            train_class_dir = os.path.join(output_train_dir, class_name)
            test_class_dir = os.path.join(output_test_dir, class_name)
            os.makedirs(train_class_dir, exist_ok=True)
            os.makedirs(test_class_dir, exist_ok=True)

            # Get list of images in the class directory
            images = os.listdir(class_dir)
            # Shuffle the images randomly
            random.shuffle(images)

            # Split images into train and test sets based on the split ratio
            split_index = int(len(images) * split_ratio)
            train_images = images[:split_index]
            test_images = images[split_index:]

            # Copy train images to train directory
            for image in train_images:
                src = os.path.join(class_dir, image)
                dst = os.path.join(train_class_dir, image)
                shutil.copyfile(src, dst)

            # Copy test images to test directory
            for image in test_images:
                src = os.path.join(class_dir, image)
                dst = os.path.join(test_class_dir, image)
                shutil.copyfile(src, dst)


# method for deterministic random split of dataset in N partitions
def split_dataset(dataset, num_partitions):
    # Get number of examples in the dataset
    num_examples = len(dataset)
    # Calculate number of examples per partition
    examples_per_partition = num_examples // num_partitions
    print(num_examples, examples_per_partition)

    # Split the dataset into partitions
    partitions = []
    for i in range(num_partitions):
        start = i * examples_per_partition
        end = start + examples_per_partition
        partition = dataset[start:end]
        partitions.append(partition)

    return partitions

def set_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    import numpy as np
    np.random.seed(seed)

def split_dataset(dataset, n_splits, seed=42):
    # Set the random seed for reproducibility
    set_random_seed(seed)
    
    # Calculate split sizes
    total_size = len(dataset)
    split_sizes = [total_size // n_splits] * n_splits
    for i in range(total_size % n_splits):
        split_sizes[i] += 1
    
    # Split the dataset
    splits = torch.utils.data.random_split(dataset, split_sizes)
    return splits

class VehicleImageDataset(Dataset):
    def __init__(self, img_labels, transform=None):
        """
        Args:
            img_labels (list of tuples): List containing (img_path, target_label).
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.img_labels = img_labels
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path, label = self.img_labels[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_partitions(
        local_train_path:str,
        local_val_path: str,
        poisoned_train_path: str,
        poisoned_val_path: str,
        num_clients: int,
        num_malicious_clients: int,
        batch_size: int,
        poison_rate: float,
) -> tuple[list[DataLoader], list[DataLoader]]:
    """
        This function returns a list of DataLoaders, each DataLoader corresponds to a client.
        A client can be either benign or malicious.

    """
    train_partitions = []
    val_partitions = []
    # Load the datasets
    trainset = ImageFolder(root=local_train_path, transform=transform)
    valset = ImageFolder(root=local_val_path, transform=transform)
    poisoned_trainset = ImageFolder(root=poisoned_train_path, transform=transform)
    poisoned_valset = ImageFolder(root=poisoned_val_path, transform=transform)
    # we get the indices of each partition
    train_partitions_indices = [partition.indices for partition in split_dataset(trainset, num_clients, seed=42)]
    val_partitions_indices = [partition.indices for partition in split_dataset(valset, num_clients, seed=42)]
    
    for i in range(num_clients-num_malicious_clients):
        train_samples = [trainset.imgs[idx] for idx in train_partitions_indices[i]]
        val_samples = [valset.imgs[idx] for idx in val_partitions_indices[i]]

        train_ds = VehicleImageDataset(train_samples, transform=transform)
        val_ds = VehicleImageDataset(val_samples, transform=transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        train_partitions.append(train_loader)
        val_partitions.append(val_loader)
    
    for i in range(num_clients-num_malicious_clients, num_clients):

        # calculate probability poison_rate to use normal sample or poisoned sample
        
        
        poisoned_train_samples = [poisoned_trainset.imgs[idx] for idx in train_partitions_indices[i]]
        poisoned_val_samples = [poisoned_valset.imgs[idx] for idx in val_partitions_indices[i]]
        normal_train_samples = [trainset.imgs[idx] for idx in train_partitions_indices[i]]
        normal_val_samples = [valset.imgs[idx] for idx in val_partitions_indices[i]]
        train_samples = []
        val_samples = []

        for j in range(len(normal_train_samples)):
            if random.random() <= poison_rate:
                path, label = poisoned_train_samples[j]
                # flip the label
                sample = (path,(int(label)+1)%2)
                train_samples.append(sample)
            else:
                train_samples.append(normal_train_samples[j])

        for j in range(len(normal_val_samples)):
            if random.random() <= poison_rate:
                path, label = poisoned_val_samples[j]
                # flip the label
                sample = (path,(int(label)+1)%2)
                val_samples.append(sample)
            else:
                val_samples.append(normal_val_samples[j])

        train_ds = VehicleImageDataset(train_samples, transform=transform)
        val_ds = VehicleImageDataset(val_samples, transform=transform)

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        train_partitions.append(train_loader)
        val_partitions.append(val_loader)
    

    return train_partitions, val_partitions

if __name__ == '__main__':
    # Example usage:
    input_directory = "./og_vehicles/"
    output_train_directory = "./local_vehicles/train"
    output_test_directory = "./local_vehicles/test"
    split_ratio = 0.7  # 70% for training, 30% for testing

    #split_images(input_directory, output_train_directory,
    #             output_test_directory, split_ratio)
