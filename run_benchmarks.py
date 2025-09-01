import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2 as transforms
from tqdm import tqdm

import random
import wandb
import numpy as np
import yaml
import os

from Custom_VIT import (
    ViTWithDecompSequenceGrading,
    ViTWithStaticPositionalEncoding,
    ViTWithAggPositionalEncoding_PF,
    ViTWithAggPositionalEncoding_SP,
    ViTWithAggPositionalEncoding_RandNoise
      )

from helpers import make_cached_loader, prepare_cached_datasets
import warnings
warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "true"

#TODO: This branch is to fix bugs in forward pass of Vanilla ViT Baseline

def get_cifar10_loaders(batch_size=128,data_dir='./data'):


    # CIFAR-10 mean and std for normalization
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Training transforms with augmentation and resize to 224x224
    # TODO: Use Deterministic Horizontal Flip from paper, and may be extend to Deterministic Affine Transform as well.
    train_transform = transforms.Compose([
        transforms.ToImage(), # PIL -> Tensor fast path
        transforms.Resize((224, 224), interpolation = transforms.InterpolationMode.BILINEAR),  # Resize to ViT's expected input size
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Test transforms with resize to 224x224
    test_transform = transforms.Compose([
        transforms.ToImage(), # PIL -> Tensor fast path
        transforms.Resize((224, 224), interpolation = transforms.InterpolationMode.BILINEAR),  # Resize to ViT's expected input size
        transforms.ToDtype(torch.float32, scale=True),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # Load datasets
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    if torch.cuda.is_available():
        num_workers = min(16, os.cpu_count() or 8); pin_memory = True
        persistent_workers = True; prefetch_factor = 4; drop_last = False
    else:
        num_workers = 2; pin_memory = False; persistent_workers = False
        prefetch_factor = 1; drop_last = True

    # Create data loaders
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=persistent_workers,
                              prefetch_factor = prefetch_factor,
                              drop_last = drop_last
                              )
    
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=num_workers,
                              pin_memory=pin_memory,
                              persistent_workers=persistent_workers,
                              prefetch_factor = prefetch_factor,
                              drop_last = drop_last
                             )

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


def batch_inference_template(model, data_loader, criterion, device):

    """ """
    # Test the model
    model.eval()

    losses = []; accuracies = []

    for index, (image_batch, label_batch) in tqdm( enumerate( iter(data_loader) ) , desc="Processing batches", total = len(data_loader)):

        # cast to bfloat16 and perform inference on GPU with channels placed last in memory
        if torch.cuda.is_available():
            image_batch = image_batch.to(device, non_blocking = True, memory_format=torch.channels_last)
            label_batch = label_batch.to(device,non_blocking = True)

            # Though using bfloat16 during inference is optional, recommended to speed up inference on GPUs that support it.
            with torch.no_grad(), torch.amp.autocast(dtype = torch.bfloat16, device_type = "cuda"):
                outputs = model(image_batch)
                loss = criterion(outputs, label_batch)
                
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == label_batch).float().mean().item()

                losses.append(loss.item())
                accuracies.append(accuracy)

        # perform regular inference on CPU
        else:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            with torch.no_grad():

                outputs = model(image_batch)
                loss = criterion(outputs, label_batch)
                
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == label_batch).float().mean().item()

                losses.append(loss.item())
                accuracies.append(accuracy)
        

    batch_loss = sum(losses) / len(losses)
    batch_acc = sum(accuracies) / len(accuracies)

    return (batch_loss, batch_acc)


def test_model(model, test_loader, CELoss, device):
    print("Testing model on test dataset")
    test_loss, test_acc = batch_inference_template(model = model, data_loader = test_loader, criterion = CELoss, device = device)
    return (test_loss, test_acc)

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
    print("Validating model on validation dataset")
    val_loss, val_acc = batch_inference_template(model = model, data_loader = val_loader, criterion = CELoss, device = device)
    return (val_loss, val_acc)

def train_model(model, train_loader, optimizer, scheduler, CELoss, device, val_model = False, val_loader = None):
    
    batch_losses = [ ]
    model.train()

    for index, (image_batch, label_batch) in tqdm( enumerate( iter(train_loader) ) , total = len(train_loader), desc = "Processing train batches"):

        # by default data should be in float32
        image_batch = image_batch.to(device, dtype = torch.float32)
        label_batch = label_batch.to(device)

        # cast ot bfloat16 and perform inference on GPU with channels placed last in memory
        if torch.cuda.is_available():
            image_batch = image_batch.to(device, non_blocking = True, memory_format=torch.channels_last)
            label_batch = label_batch.to(device,non_blocking = True)

            with torch.amp.autocast(dtype = torch.bfloat16, device_type = "cuda"):
                # Forward pass through the model through optimized feature representation
                outputs = model(image_batch)
        
        # perform regular inference on CPU
        else:
            outputs = model(image_batch)


        optimizer.zero_grad()
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


def setup_training(data_paths, num_epochs, model, device,
                   batch_size = 512,
                   patience=16, min_delta_loss=1e-8, min_epochs=20, smooth_k=7, run_logger = None,local_testing=False):
    
    train_loader = make_cached_loader(data_paths['train_data'], batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = make_cached_loader(data_paths['val_data'], batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = make_cached_loader(data_paths['test_data'], batch_size=batch_size, shuffle=False, num_workers=4)

    # only include trainable params to optimizer. Need to rebuild or add a new param group if adding new layers later
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    val_accs, val_losses, best_val_loss_sm = [],[], float('inf')
    epochs_no_improve, ckpt_path = 0, 'best_by_valloss.pth'
    
    # by default model should be in float 32
    model = model.to(device, dtype = torch.float32)

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True
        model = model.to(device, memory_format=torch.channels_last)

    criterion = torch.nn.CrossEntropyLoss()

    for ep in tqdm( range(num_epochs) , desc = "Running epochs for training"):

        if ep % 2 == 0:
            train_loss, val_loss, val_acc = train_model(model,
                                                        train_loader,
                                                        optimizer,
                                                        scheduler,
                                                        criterion,
                                                        device,
                                                        val_model=True,
                                                        val_loader=val_loader
                                                        )
        
            val_accs.append(val_acc)
            val_losses.append(val_loss)
        
            loss_window = val_losses[-smooth_k:] if len(val_losses) >= smooth_k else val_losses
            val_loss_sm = float(np.mean(loss_window))

            # Saving best model validation loss and a corresponding checkpoint
            if val_loss_sm < (best_val_loss_sm - min_delta_loss):
                best_val_loss_sm = val_loss_sm
                epochs_no_improve = 0
                torch.save({'epoch': ep,
                            'model_state': model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                            'val_loss_sm': val_loss_sm,
                            'val_acc': val_acc},
                            ckpt_path)
            
            # Terminating training because epoch didn't improve until patience time
            else:
                epochs_no_improve += 1
                if (ep + 1) >= min_epochs and epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {ep}."); break

            if not local_testing:
                # Log only in remote experiments, not local
                run_logger.log({"train_loss": train_loss,
                                "val_loss": val_loss,
                                "val_accuracy": val_acc
                                })
        
        # Logging only train metrics to wandb
        else:
            train_loss = train_model(model,
                                     train_loader,
                                     optimizer,
                                     scheduler,
                                     criterion,
                                     device,
                                     val_model=False
                                     )
            # Log only in remote experiments, not local
            if not local_testing: run_logger.log({"train_loss": train_loss})

    checkpoint = torch.load(ckpt_path, map_location=device)
    print(f"Best model in terms of validation loss was at: {checkpoint['epoch']+1} epoch, loading it for testing.")
    model.load_state_dict(checkpoint['model_state'])
    test_loss, test_acc = test_model(model, test_loader, criterion, device)
    
    if not local_testing:
        run_logger.log({"test_accuracy": test_acc, "test_loss": test_loss}); run_logger.finish()
    
    return (test_loss, test_acc)



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

    yaml_project_name = "aggregate_pos_enc_FiLMInj"
    log_metrics = True

    config_details = get_project_details("./configs.yaml", yaml_project_name)
    set_system_seed(config_details['config']['system_seed'])
    
    model = get_model(model_name = config_details['model_name'],
                      model_config = config_details['config']
                      )
    device = get_device()

    data_paths = prepare_cached_datasets('./data/cache_gpu/')

    if log_metrics:
        run_logger = init_wandb(team_name=config_details['team_name'],
                    project_name=config_details['project_name'],
                    run_name = config_details['run_name'] + '_' + str(config_details['config']['system_seed'] ),
                    secret_key=config_details['secret_key'],
                    additional_config = config_details['config']
                    )
        
        run_logger.log_code("./")

    test_loss, test_acc = setup_training(data_paths = data_paths,
                   num_epochs = config_details['config']['num_epochs'],
                   model = model,
                   device=device,
                   batch_size = 512,
                   patience=10,
                   min_delta_loss=1e-8,
                   min_epochs=20,
                   smooth_k=3,
                   run_logger = run_logger if log_metrics else None,
                   local_testing= not log_metrics)
    
    print("Test Loss: ", test_loss)
    print("Test accuracy: ", test_acc)
