import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

import random
import wandb
import numpy as np
import yaml

from Custom_VIT import (
    ViTWithDecompSequenceGrading,
    ViTWithStaticPositionalEncoding,
    ViTWithAggPositionalEncoding_PF,
    ViTWithAggPositionalEncoding_SP,
    ViTWithAggPositionalEncoding_RandNoise
     
      )



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

        settings = wandb.Settings(code_dir = './')
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


def validate_model(model, val_loader, CELoss, device):
    
    model.eval()
    with torch.no_grad():

        total_loss = 0; total_acc = 0
        num_batches = len(val_loader)
        
        for index, (batch_data, batch_labels) in tqdm( enumerate( iter(val_loader) ), total = num_batches, desc = "processing val batches"):

            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_data)
            loss = CELoss(outputs, batch_labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            accuracy = (predicted == batch_labels).float().mean().item()
            total_acc += accuracy
        
        avg_val_loss = total_loss / num_batches
        avg_val_acc = total_acc / num_batches


    return (avg_val_loss, avg_val_acc)


def train_model(model, train_loader, optimizer, scheduler, CELoss, device, val_model = False, val_loader = None):
    
    batch_losses = [ ]
    model = model.to(device)

    model.train()

    for index, (image_batch, label_batch) in tqdm( enumerate( iter(train_loader) ) , total = len(train_loader), desc = "Processing train batches"):

        image_batch = image_batch.to(device)
        label_batch = label_batch.to(device)

        optimizer.zero_grad()

        # Forward pass through the model
        outputs = model(image_batch)
        loss = CELoss(outputs, label_batch)

        batch_losses.append(loss.item())
        
        loss.backward()
        optimizer.step()

        

    train_losses = sum(batch_losses)/len(batch_losses)
    scheduler.step()

    if val_model:
        val_loss, val_acc = validate_model(model, val_loader, CELoss, device)
        return (train_losses, val_loss, val_acc)

    else: return train_losses


def setup_training(num_epochs, model, run_logger, device,
                   patience=7, min_delta_loss=1e-8, min_epochs=20, smooth_k=3):
    
    train_loader, test_loader = get_cifar10_loaders(batch_size=64, num_workers=2)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    val_accs, val_losses, best_val_loss_sm = [],[], float('inf')
    epochs_no_improve, ckpt_path = 0, 'best_by_valloss.pth'
    val_model = True; val_loader = None

    if val_model:
        total_tr_rows = len(train_loader.dataset)
        tr_rows = int(0.7 * total_tr_rows)
        val_rows = int(total_tr_rows - tr_rows)
        
        train_loader, val_loader = get_val_splits(train_dataset = train_loader.dataset,
                    tr_size = tr_rows,
                    val_size = val_rows,
                    tr_bs = 64,
                    val_bs = 32)

    
    criterion = torch.nn.CrossEntropyLoss()

    for ep in range(num_epochs):
        train_loss, val_loss, val_acc = train_model(model,
                                                    train_loader,
                                                    optimizer,
                                                    scheduler,
                                                    criterion,
                                                    device,
                                                    val_model=val_model,
                                                    val_loader=val_loader
                                                    )
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        
        loss_window = val_losses[-smooth_k:] if len(val_losses) >= smooth_k else val_losses
        val_loss_sm = float(np.mean(loss_window))

        if val_loss_sm < (best_val_loss_sm - min_delta_loss):
            best_val_loss_sm = val_loss_sm
            epochs_no_improve = 0
            torch.save({'epoch': ep,
                        'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'val_loss_sm': val_loss_sm,
                        'val_acc': val_acc},
                        ckpt_path)
        else:
            epochs_no_improve += 1
            if (ep + 1) >= min_epochs and epochs_no_improve >= patience:
                print(f"Early stopping at epoch {ep}."); break

        run_logger.log({"train_loss": train_loss,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "best_val_loss_smoothed": best_val_loss_sm
                        })

    model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state'])
    test_loss, test_acc = test_model(model, test_loader, criterion, device)
    run_logger.log({"test_accuracy": test_acc, "test_loss": test_loss}); run_logger.finish()



def get_model(model_name, model_config):

    if model_name == 'decomp':
        
        model = ViTWithDecompSequenceGrading(
            pretrained_model_name="google/vit-base-patch16-224",
            decomp_algo=model_config['algo'],  # or qr or svd
            decomp_strategy=model_config['strategy'],  # project or importance
            top_k_seqfeat = model_config["top_k_seqfeat"],
            num_out_classes=model_config['num_out'],
            alpha = model_config['alpha']
                                            )
    
    if model_name == 'aggregate_pf':
        model = ViTWithAggPositionalEncoding_PF(
            distance_metric=model_config['distance_metric'],
            aggregate_method=model_config['aggregate_method'],
            seq_select_method = model_config['seq_select_method'],
            num_out_classes = model_config['num_out'],
            aggregate_dim = model_config['aggregate_dim'],
            norm_type = model_config['norm_type'],
            return_anchors = model_config['return_anchors'],
            use_both=model_config['use_both']
        )

    if model_name == 'aggregate_pos_enc_FiLMInj':
        model = ViTWithAggPositionalEncoding_SP(
            distance_metric=model_config['distance_metric'],
            aggregate_method=model_config['aggregate_method'],
            seq_select_method = model_config['seq_select_method'],
            num_out_classes = model_config['num_out'],
            add_coordinates = model_config['add_coordinates'],
            K = model_config['K'],
            aggregate_dim = model_config['aggregate_dim'],
            norm_type = model_config['norm_type'],
            return_anchors = model_config['return_anchors']
                                        )
    
    if model_name == 'aggregate_pos_enc_FiLMInj_random':

        model = ViTWithAggPositionalEncoding_RandNoise(
            num_out_classes=model_config['num_out']
        )


    if model_name == 'static':
        
        model = ViTWithStaticPositionalEncoding(
            num_out_classes=model_config['num_out']
        )
    
    return model

def get_project_details(yaml_config_file, exp_name):

    with open(yaml_config_file, 'r') as file:
        loaded_config = yaml.safe_load(file)
    
    # loaded_config - 'yaml_project_name' , config, model_name, team_name, project_name, run_name, secret_key
    if exp_name in loaded_config:
        return loaded_config[exp_name]
    else:
        raise Exception("Provided experiment doesn't exist")


if __name__ == "__main__":

    yaml_project_name = "static_pos_enc"

    config_details = get_project_details("./configs.yaml", yaml_project_name)
    set_system_seed(config_details['config']['system_seed'])
    
    model = get_model(model_name = config_details['model_name'],
                      model_config = config_details['config']
                      )
    device = get_device()

    run_logger = init_wandb(team_name=config_details['team_name'],
                project_name=config_details['project_name'],
                run_name = config_details['run_name'] + '_' + str(config_details['config']['system_seed'] ),
                secret_key=config_details['secret_key'],
                additional_config = config_details['config']
                )
    
    run_logger.log_code("./")

    setup_training(config_details['config']['num_epochs'], model, run_logger, device)
