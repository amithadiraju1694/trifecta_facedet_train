import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2 as transforms

import random
import wandb
import numpy as np

from Custom_VIT import ViTWithAggPositionalEncoding, ViTWithDecompPositionalEncoding, ViTWithStaticPositionalEncoding



def get_cifar10_loaders(batch_size=128, num_workers=4, data_dir='./data'):
    # CIFAR-10 mean and std for normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Training transforms with augmentation and resize to 224x224
    # TODO: Use Deterministic Horizontal Flip from paper, and may be extend to Deterministic Affine Transform as well.
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to ViT's expected input size
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Test transforms with resize to 224x224
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize to ViT's expected input size
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load datasets
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)

    return train_loader, test_loader


def init_wandb(team_name, project_name, run_name, secret_key, additional_config = None):

    try:
        wandb.login(key = secret_key)
    except:
        raise Exception("Error logging into Wandb with provided token")

    # Start a new wandb run to track this script.
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity=team_name,

        # Set the wandb project where this run will be logged.
        project=project_name,

        name = run_name,

        config = additional_config,

        settings = wandb.Settings(code_dir = '.')
    )

    return run

def get_device():

    device = torch.device('cpu')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    
    # if torch.mps.is_available():
    #     device = torch.device('mps') # This was hanging system
    
    print("Using device: ", device)
    

    return device


def test_model(model, test_loader, CELoss, device):

    model = model.to(device)
    # Test the model
    model.eval()

    bloss = []; bacc = []

    for index, (image_batch, label_batch) in enumerate( iter(test_loader) ):

        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        if index > 2:
            break

        with torch.no_grad():

            test_outputs = model(image_batch)
            test_loss = CELoss(test_outputs, label_batch)
            
            _, predicted = torch.max(test_outputs, 1)
            accuracy = (predicted == label_batch).float().mean().item()

            bloss.append(test_loss)
            bacc.append(accuracy)
    

    batch_loss = sum(bloss) / len(bloss)
    batch_acc = sum(bacc) / len(bacc)

    return (batch_loss, batch_acc)


def set_system_seed(seed_num):
    torch.manual_seed(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)


def get_val_splits(train_dataset, tr_size, val_size, tr_bs, val_bs):

    tr_dt, val_dt = random_split(train_dataset, [tr_size, val_size])

    train_loader = DataLoader(tr_dt, batch_size = tr_bs)
    val_loader = DataLoader(val_dt, batch_size = val_bs)

    return(train_loader, val_loader)


def validate_model(model):
    pass


#TODO : Finish validating model and 
def train_model(model, train_loader, optimizer, scheduler, CELoss, device):
    
    batch_losses = [ ]; batch_acc = []
    model = model.to(device)

    model.train()

    # total_tr_rows = len(train_loader)
    # tr_rows = int(0.7 * total_tr_rows)
    
    # train_loader, val_loader = get_val_splits(train_dataset = train_loader.dataset,
    #                tr_size = tr_rows,
    #                val_size = total_tr_rows - tr_rows,
    #                tr_bs = 64,
    #                val_bs = 32)

    for index, (image_batch, label_batch) in enumerate( iter(train_loader) ):

        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad()

        # Forward pass through the model
        outputs = model(image_batch)

        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == label_batch).float().mean().item()
        loss = CELoss(outputs, label_batch)

        batch_losses.append(loss.item())
        batch_acc.append(accuracy)
        
        loss.backward()
        optimizer.step()

        # Remove this, just for testing
        if index > 2:
            print("Breaking from batches")
            break

    all_batch_losses = sum(batch_losses)/len(batch_losses)
    all_batch_acc = sum(batch_acc)/len(batch_acc)
    
    
    scheduler.step()

    return (all_batch_losses, all_batch_acc)


def setup_training(num_epochs, model, run_logger, device):

    train_loader, test_loader = get_cifar10_loaders(batch_size=64, num_workers=2)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # 2. Learning Rate Scheduler: Cosine Annealing with Warm Restarts
    # T_0 is the number of epochs for the first restart cycle; T_mult is a factor for increasing the cycle length after each restart
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                                     T_0=10,
                                                                     T_mult=1,
                                                                     eta_min=1e-6)

    CELoss = torch.nn.CrossEntropyLoss()

    losses = []; accs = []
    for ep in range(num_epochs):

        loss, accuracy = train_model(model, train_loader,optimizer, scheduler, CELoss, device)
        
        if ep % 20 == 0:
            print(f"Loss of : {loss} , accuracy of : {accuracy} in epoch: {ep}")

        losses.append(loss)
        accs.append(accuracy)

        run_logger.log(
            {"train_loss": loss,
             "train_accuracy": accuracy
             }
        )

    
    test_loss, test_acc = test_model(model, test_loader, CELoss, device)

    run_logger.log({
                    "test_accuracy": test_acc,
                    "test_loss" : test_loss
                            })

    run_logger.finish()

    return


def get_model(model_name, model_config):

    if model_name == 'decomp':
        
        model = ViTWithDecompPositionalEncoding(
            pretrained_model_name="google/vit-base-patch16-224",
            decomp_algo=model_config['algo'],  # or qr or svd
            decomp_strategy=model_config['strategy'],  # project or importance
            num_out_classes=model_config['num_out'],
            alpha = model_config['alpha']
                                                )

    if model_name == 'aggregate':

        model = ViTWithAggPositionalEncoding(
            distance_metric=model_config['distance'], # cosine, euclidean
            aggregate_method=model_config['agg'], # max_elem
            alpha= model_config['alpha'],
            num_out_classes=model_config['num_out']
        )
    
    if model_name == 'static':
        
        model = ViTWithStaticPositionalEncoding(
            num_out_classes=model_config['num_out']
        )
    
    return model


if __name__ == "__main__":

    system_seed = 108
    num_epochs = 5
    set_system_seed(system_seed)

    config_decomp = {
        'algo': 'eig',  # or qr or svd
        'strategy': 'importance',  # project or importance
        'num_out': 10,
        'alpha' : 0.7,
        'system_seed':system_seed,
        'num_epochs': num_epochs
                    }
    
    config_agg = {'distance': 'cosine', # cosine, euclidean
     'agg': 'max_elem', # max_elem
      'alpha': 0.7,
    'num_out': 10,
     'system_seed': system_seed,
     'num_epochs': num_epochs
      }
    
    config_static = {'num_out': 10,
                     'system_seed': system_seed,
                     'num_epochs': num_epochs
                     }

    
    
    model = get_model(model_name = 'decomp', model_config = config_decomp)
    device = get_device()

    run_logger = init_wandb(team_name='amith-adiraju-self',
                project_name='hybrid_posenc_test_v2',
                run_name = 'decomp_pos_enc',
                secret_key='f07b2137ee2ba424d6b068954595c97b1f669138',
                additional_config = config_decomp
                )
    
    run_logger.log_code("Project Code")

    setup_training(num_epochs, model, run_logger, device)
