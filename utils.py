from efficientkan import KAN

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, random_split
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

import matplotlib.pyplot as plt
import math
import pandas as pd


class MissingDatasetException(Exception):
    pass


class MyMLP(nn.Module):

    def __init__(self, layers: list):
        super(MyMLP, self).__init__()

        self.layers = layers
        to_sequential, self.num_parameters = self.build_layers_and_count_params(layers)
        self.build_layers = nn.Sequential(*to_sequential)

    def forward(self, x):
        return self.build_layers(x)


    # given a list of layer sizes, builds the respective linear layers with ReLU activation functions.
    # Also returns the total number of parameters of the model.
    def build_layers_and_count_params(self, layers):

        to_sequential = []
        num_parameters = 0

        for i in range(len(layers) - 1):

            to_sequential.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                to_sequential.append(nn.ReLU())

            num_parameters += layers[i]*layers[i+1] + layers[i+1]

        return to_sequential, num_parameters


class MyKAN(nn.Module):

    def __init__(self, layers: list, grid_size):
        super(MyKAN, self).__init__()

        self.layers = layers
        self.num_parameters = self.count_params(layers, grid_size)
        self.build_layers = KAN(layers, grid_size=grid_size)


    def forward(self, x):
        return self.build_layers(x)


    def count_params(self, layers, grid_size):

        num_parameters = 0

        for i in range(len(layers) - 1):
            num_parameters += layers[i]*layers[i+1]

        return num_parameters * grid_size


# ================================ IMAGE CLASSIFICATION ================================

def get_img_dataset(ds_name="MNIST", batch_size=256):
    '''
    Initiates one of four datasets: MNIST, FMNIST, CIFAR10, CIFAR100.
    '''
    
    if ds_name not in ["MNIST", "FMNIST", "CIFAR10", "CIFAR100"]:
        raise MissingDatasetException(f'Cannot parse dataset "{ds_name}". Available datasets: "MNIST", "FMNIST", "CIFAR10", "CIFAR100".')


    elif ds_name == "MNIST":

        transform = transforms.Compose([
        #torchvision.transforms.Resize(14),      # Resize the image
        transforms.ToTensor(),                  # Convert images to PyTorch tensors and scale to [0,1]
        transforms.Normalize((0.1307,), (0.3081,))    # Normalize to mean=0.5, std=0.5
        ])

        trainset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        testset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

        input_size = 1 * 28 * 28
        num_classes = 10

    elif ds_name == "FMNIST":

        transform = transforms.Compose([
        #torchvision.transforms.Resize(14),      # Resize the image
        transforms.ToTensor(),                  # Convert images to PyTorch tensors and scale to [0,1]
        transforms.Normalize((0.5,), (0.5,))    # Normalize to mean=0.5, std=0.5
        ])

        trainset = datasets.FashionMNIST(root="./data", train=True, download=True, transform=transform)
        testset = datasets.FashionMNIST(root="./data", train=False, download=True, transform=transform)

        input_size = 1 * 28 * 28
        num_classes = 10

    elif ds_name == "CIFAR10":

        transform = transforms.Compose([
        #torchvision.transforms.Resize(14),      # Resize the image
        transforms.ToTensor(),                  # Convert images to PyTorch tensors and scale to [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))    # Normalize to mean=0.5, std=0.5
        ])

        trainset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)

        input_size = 3 * 32 * 32
        num_classes = 10

    elif ds_name == "CIFAR100":

        transform = transforms.Compose([
        #torchvision.transforms.Resize(14),      # Resize the image
        transforms.ToTensor(),                  # Convert images to PyTorch tensors and scale to [0,1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))     # Normalize to mean=0.5, std=0.5
        ])

        trainset = datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
        testset = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)


        input_size = 3 * 32 * 32
        num_classes = 100

    trainset_to_display = trainset

    # Dividing into train set and validation set
    train_size = int(0.8 * len(trainset))
    val_size = len(trainset) - train_size
    trainset, valset = random_split(trainset, [train_size, val_size])

    # Creating data loaders
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    # Printing data shapes
    '''print("Single image dimensions:", trainset.data[0].shape)
    print("Single Image resolution:", input_size)
    for images, labels in trainloader:
        print("Image batch dimensions:", images.shape)
        print("Image label dimensions:", labels.shape)
        break'''
        
    return trainset_to_display, trainloader, valloader, testloader, input_size, num_classes


def display_plots_epochs_old(model_list, total_train_losses, total_test_losses, total_train_accs, total_test_accs, num_epochs):
    '''
    Displays train/test loss and accuracy w.r.t epochs (verbose).
    '''

    fig, ax = plt.subplots(3,4, figsize=(32,18))
    fig.text(0.5, 0.06, 'Epochs', ha='center', va='center', fontsize=14)

    for i, model in enumerate(model_list):

        row_loss = math.floor( (i*2) / 4)
        col_loss = i*2 % 4

        row_acc = math.floor( ((i*2)+1) / 4)
        col_acc = ((i*2)+1) % 4

        ax[row_loss,col_loss].plot(range(1, num_epochs+1), total_train_losses[i], label='Train loss')
        ax[row_loss,col_loss].plot(range(1, num_epochs+1), total_test_losses[i], label='Test loss')
        ax[row_loss,col_loss].legend()
        ax[row_loss,col_loss].set_title(f'{model.__class__.__name__} {model.layers}')
        ax[row_loss,col_loss].set_xlabel('Epochs')
        ax[row_loss,col_loss].set_ylabel('Loss')

        ax[row_acc,col_acc].plot(range(1, num_epochs+1), total_train_accs[i], label='Train accuracy')
        ax[row_acc,col_acc].plot(range(1, num_epochs+1), total_test_accs[i], label='Test accuracy')
        ax[row_acc,col_acc].legend()
        ax[row_acc,col_acc].set_title(f'{model.__class__.__name__} {model.layers}')
        ax[row_acc,col_acc].set_xlabel('Epochs')
        ax[row_acc,col_acc].set_ylabel('Accuracy')


def display_plots_epochs(ds_name, model_list, total_train_losses, total_val_losses, total_train_accs, total_val_accs, num_epochs):
    '''
    Displays train/validation loss and accuracy w.r.t epochs.
    '''

    fig, ax = plt.subplots(1,2, figsize=(8,4))

    colors = plt.cm.tab10.colors

    for i, model in enumerate(model_list):

        base_color = colors[i % len(colors)]
        
        ax[0].plot(range(1, num_epochs + 1), total_train_losses[i], color=base_color, alpha=0.33)
        ax[0].plot(range(1, num_epochs + 1), total_val_losses[i], label=f'{model.__class__.__name__}{model.layers}', color=base_color)
        
        ax[1].plot(range(1, num_epochs + 1), total_train_accs[i], color=base_color, alpha=0.33)
        ax[1].plot(range(1, num_epochs + 1), total_val_accs[i], label=f'{model.__class__.__name__}{model.layers}', color=base_color)

    ax[0].set_xlabel('Epoch')
    ax[0].set_ylabel('Loss')
    ax[0].set_title(f'{ds_name} - Train and Valid. Loss')
    ax[0].legend()

    ax[1].set_xlabel('Epoch')
    ax[1].set_ylabel('Accuracy')
    ax[1].set_title(f'{ds_name} - Train and Valid. Accuracy')
    ax[1].legend()

    plt.tight_layout()
    plt.show()


def sort_by_params(*args: list):
    '''
    Sorts the first list and then the others accordingly. list1=[3,1,2], list2=[c,a,b] --> list1=[1,2,3], list2=[a,b,c]
    '''

    sorted_lists = sorted(zip(*args))
    sorted_lists = zip(*sorted_lists)
    sorted_lists = [list(lst) for lst in sorted_lists]

    return sorted_lists


def shallow_or_deep(models_layers):
    '''
    Tells if a model has one (S) or more (D) hidden layers.
    '''

    if len(models_layers) == 3:
        return '$S$'
    else:
        return '$D$'


def display_plots_params(ds_name, test_accs_mlp, test_accs_kan, num_params_mlp, num_params_kan):
    '''
    Displays test accuracy w.r.t. number of parameters.
    '''

    num_params_mlp, test_accs_mlp = sort_by_params(num_params_mlp, test_accs_mlp)
    num_params_kan, test_accs_kan = sort_by_params(num_params_kan, test_accs_kan)

    print(test_accs_mlp)
    print(test_accs_kan)
    print(num_params_mlp)
    print(num_params_kan)

    fig, ax = plt.subplots(figsize=(3.7,3))

    ax.scatter(num_params_mlp, test_accs_mlp, color='blue', marker='o')  # MLP points
    ax.scatter(num_params_kan, test_accs_kan, color='orange', marker='o')  # KAN points

    ax.plot(num_params_mlp, test_accs_mlp, label='MLP')
    ax.plot(num_params_kan, test_accs_kan, label='KAN')

    ax.legend(loc='lower right')
    ax.set_xlabel('Number of parameters')
    ax.set_ylabel('Test accuracy')
    ax.set_title(ds_name)


def display_plots_params_enhanced(ds_name, models_layers_mlp, models_layers_kan, test_accs_mlp, test_accs_kan, num_params_mlp, num_params_kan):
    '''
    Displays test accuracy w.r.t. number of parameters, highlighting if the model is shallow (S) or deep (D).
    '''

    num_params_mlp, test_accs_mlp, models_layers_mlp = sort_by_params(num_params_mlp, test_accs_mlp, models_layers_mlp)
    num_params_kan, test_accs_kan, models_layers_kan = sort_by_params(num_params_kan, test_accs_kan, models_layers_kan)

    print(test_accs_mlp)
    print(test_accs_kan)
    print(num_params_mlp)
    print(num_params_kan)
    print(models_layers_mlp)
    print(models_layers_kan)

    fig, ax = plt.subplots(figsize=(3.5,3))

    for i in range(len(models_layers_mlp)):
        ax.scatter(num_params_mlp[i], test_accs_mlp[i], color='blue', marker=shallow_or_deep(models_layers_mlp[i]))
        ax.scatter(num_params_kan[i], test_accs_kan[i], color='orange', marker=shallow_or_deep(models_layers_kan[i]))

    ax.plot(num_params_mlp, test_accs_mlp, label='MLP', alpha=0.4)
    ax.plot(num_params_kan, test_accs_kan, label='KAN', alpha=0.4)

    ticks = [14000, 120000, 330000]
    ax.set_xticks(ticks)
    ax.set_xticklabels(['~14,000', '~120,000', '~330,000'], rotation=45, ha="right")

    ax.legend(loc='lower right')
    ax.set_xlabel('Number of parameters')
    ax.set_ylabel('Test accuracy')
    ax.set_title(ds_name)



# ================================ REGRESSION ================================

class RMSELoss(nn.Module):
    '''
    RMSE Loss. The eps=1e-6 parameter eliminates NaN problems during backward pass.
    '''

    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss


def get_regr_dataset(device, ds_id=9, batch_size=64, test_size=0.2):
    '''
    Initiates one of three datasets: Auto MPG (id=9), Airfoil self noise (id=291), Appliances energy prediction (id=374).
    '''
    
    if ds_id not in [9, 291, 374, 320, 849]:
        raise MissingDatasetException(f'Cannot parse dataset with id={ds_id}. Available datasets: Auto MPG (id=9), Airfoil self noise (id=291), Appliances energy prediction (id=374).')

    # Fetch dataset 
    ds = fetch_ucirepo(id=ds_id)

    # Transform the dataset into a pandas dataset, dropping na values
    pandas_ds = pd.concat([ds.data.features, ds.data.targets], axis=1)
    pandas_ds = pandas_ds.dropna()

    # Divide into features and target
    X = pandas_ds.drop(pandas_ds.columns[-1], axis=1)
    y = pandas_ds.drop(pandas_ds.columns[:pandas_ds.shape[1]-1], axis=1)
    #print(X.head())

    # Per-dataset data cleaning
    if ds_id == 374:
        X = X.drop(columns=['date'])

    if ds_id == 849:
        X = X.drop(columns=['DateTime', 'Zone 1 Power Consumption', 'Zone 2  Power Consumption'])

    if ds_id == 320:
        X = X.drop(columns=['G1', 'G2'])
        encoder = OrdinalEncoder()
        X[X.columns[[0,1,3,4,5,8,9,10,11,14,15,16,17,18,19,20,21,22]]] = encoder.fit_transform(X[X.columns[[0,1,3,4,5,8,9,10,11,14,15,16,17,18,19,20,21,22]]])

    # Normalize the features
    X_scaled = StandardScaler().fit_transform(X)

    # Split into train, test, validation sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Convert data to PyTorch tensors and create data loaders
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    y_test = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
    y_val = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)

    X_train, X_test, X_val = X_train.to(device), X_test.to(device), X_val.to(device)
    y_train, y_test, y_val = y_train.to(device), y_test.to(device), y_val.to(device)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False)

    input_size = X_train.shape[1]

    return X, y, train_loader, test_loader, val_loader, input_size


def get_ds_name(ds_id):
    if ds_id == 9:
        return 'Auto MPG'
    elif ds_id == 291:
        return 'Airfoil self-noise'
    elif ds_id == 374:
        return 'Appliances energy'
    elif ds_id == 320:
        return 'Student performance'
    elif ds_id == 849:
        return 'Power consumption'
    else:
        raise MissingDatasetException(f'Cannot parse dataset with id={ds_id}. Available datasets: Auto MPG (id=9), Airfoil self noise (id=291), Appliances energy prediction (id=374).')