from local_dataset import (
    get_vehicle_dataset, 
    get_poisoned_vehicle_dataset, 
    show_random_sample,
    split_dataset,
    get_partitions
)

import flwr as fl

from utils import weighted_average, load_datasets
from clients import DEVICE, get_client_fn

NUM_CLIENTS = 10
BATCH_SIZE = 32

"""
    Funcio que retorni llista de DataLoaders, cada DataLoader correspon a un client.
    Cada client ha de tenir un trainloader i un valloader.
    Els clients han de tenir dades diferents, no poden compartir dades.
    Escollir X clients de forma aleatoria i fer que siguin atacants.
    Els atacants han de tenir dades enverinades.
    Una imatge normal i una imatge enverinada tenen el mateix nom, pero estan en carpetes diferents.

    imagefolder.imgs [(path, label), ...]. Fer deterministic random split. Cada split aniria directament
    a un DataLoader. Abans de ficarho al dataloader, pillar splits que seran enverinats, agafar la seva
    contra part poisoned, i llavors si que ficarho al dataloader.
"""


if __name__ == '__main__':
    trainset, valset = get_partitions(
        local_train_path='./local_vehicles/train',
        local_val_path='./local_vehicles/test',
        poisoned_train_path='./poisoned_vehicles/train',
        poisoned_val_path='./poisoned_vehicles/test',
        num_clients=NUM_CLIENTS,
        num_malicious_clients=2,
        batch_size=BATCH_SIZE,
        poison_rate=0.0,
    )
    print(trainset, valset)
    print(trainset[0].dataset.img_labels[0])
    print(trainset[-1].dataset.img_labels[0])
#    show_random_sample(trainset[0].dataset)
#    show_random_sample(trainset[-1].dataset)

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
        client_fn=get_client_fn(trainset, valset),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
        client_resources=client_resources,
    )
