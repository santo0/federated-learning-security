'''
Code from https://flower.ai/docs/framework/tutorial-series-get-started-with-flower-pytorch.html
'''

from utils import load_datasets, test, train, Net, DEVICE


if __name__ == '__main__':
    trainloaders, valloaders, testloader = load_datasets()

    batch = next(iter(trainloaders[0]))
    images, labels = batch["img"], batch["label"]

    trainloader = trainloaders[0]
    valloader = valloaders[0]
    net = Net().to(DEVICE)

    for epoch in range(5):
        train(net, trainloader, 1)
        loss, accuracy = test(net, valloader)
        print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

    loss, accuracy = test(net, testloader)
    print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")
