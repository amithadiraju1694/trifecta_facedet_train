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
from typing import Tuple


from Custom_VIT import (
    ViTWithStaticPositionalEncoding,
    ViTRADAR_SoftDegrade,
    ViTRADAR_SoftAnchor_v1,
    ViTWithPEG
      )

from helpers import make_cached_loader, prepare_cached_datasets, profile_models
import warnings
warnings.filterwarnings("ignore")
os.environ["WANDB_SILENT"] = "true"


def init_wandb(team_name: str, project_name: str, run_name:str, secret_key:str, additional_config: dict = None):

    """" Function that initializes a WanDB session with the provided parameters."""

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
    
    print("Using device: ", device)
    

    return device


def topk_acc(logits, labels, ks=(1,5)):
    maxk = max(ks)
    # logits: (B, C), labels: (B,) long
    _, topk_idx = logits.topk(maxk, dim=1, largest=True, sorted=True)  # (B, maxk)
    topk_idx = topk_idx.transpose(0,1)  # (maxk, B)
    correct = topk_idx.eq(labels.view(1, -1))  # (maxk, B)

    res = {}
    for k in ks:
        acc = correct[:k].any(dim=0).float().mean().item()
        res[f"top{k}"] = acc
    return res


def batch_inference_template_topN(model, data_loader, criterion, device, topN_tup = (1,5)) -> Tuple[float, float]:

    """
    Function that serves as template for both validation and test sets of model, a template to avoid code duplication.

    Args:
        model: The neural network model to be evaluated.
        data_loader: DataLoader object providing batches of data for evaluation.
        criterion: Loss function used to compute the loss.
        device: The device (CPU or GPU) on which computations will be performed.
        topN_tup: A tuple of integers specifying which top-N accuracies to compute.
    Returns:
        A tuple containing the average loss and accuracy over the dataset.
    """
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
                outputs = model(pixel_values = image_batch, train_mode = False)
                loss = criterion(outputs, label_batch)
                
                topk_dict = topk_acc(outputs, label_batch, ks=topN_tup)

                losses.append(loss.item())
                accuracies.append(topk_dict)

        # perform regular inference on CPU
        else:
            image_batch = image_batch.to(device)
            label_batch = label_batch.to(device)

            with torch.no_grad():

                outputs = model(pixel_values = image_batch, train_mode = False)
                loss = criterion(outputs, label_batch)
                
                topk_dict = topk_acc(outputs, label_batch, ks=topN_tup)

                losses.append(loss.item())
                accuracies.append(topk_dict)
        

    batch_loss = sum(losses) / len(losses)
    batch_accuracies = {
        f"top{k}": sum(acc[f"top{k}"] for acc in accuracies) / len(accuracies)
        for k in topN_tup
    }

    return (batch_loss, batch_accuracies)


def batch_inference_template(model, data_loader, criterion, device) -> Tuple[float, float]:

    """
    Function that serves as template for both validation and test sets of model, a template to avoid code duplication.

    Args:
        model: The neural network model to be evaluated.
        data_loader: DataLoader object providing batches of data for evaluation.
        criterion: Loss function used to compute the loss.
        device: The device (CPU or GPU) on which computations will be performed.
    Returns:
        A tuple containing the average loss and accuracy over the dataset.
    """
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
                outputs = model(pixel_values = image_batch, train_mode = False)
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

                outputs = model(pixel_values = image_batch, train_mode = False)
                loss = criterion(outputs, label_batch)
                
                _, predicted = torch.max(outputs, 1)
                accuracy = (predicted == label_batch).float().mean().item()

                losses.append(loss.item())
                accuracies.append(accuracy)
        

    batch_loss = sum(losses) / len(losses)
    batch_acc = sum(accuracies) / len(accuracies)

    return (batch_loss, batch_acc)


def test_model(model, test_loader, loss_function, device, topN = False, topN_tup = None):
    print("Testing model on test dataset")
    if not topN:
        test_loss, test_acc = batch_inference_template(model = model, data_loader = test_loader, criterion = loss_function, device = device)
    else:
        assert topN_tup is not None, "Please provide a tuple of topN values to compute accuracies for"
        test_loss, test_acc = batch_inference_template_topN(model = model, data_loader = test_loader, criterion = loss_function, device = device, topN_tup = topN_tup)
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


def validate_model(model, val_loader, loss_function, device, topN = False, topN_tup = None):
    print("Validating model on validation dataset")
    if not topN:
        val_loss, val_acc = batch_inference_template(model = model, data_loader = val_loader, criterion = loss_function, device = device)
    
    else:
        assert topN_tup is not None, "Please provide a tuple of topN values to compute accuracies for"
        val_loss, val_acc = batch_inference_template_topN(model = model, data_loader = val_loader, criterion = loss_function, device = device, topN_tup = topN_tup)
    return (val_loss, val_acc)

def log_model_to_wandb(run_logger, ckpt_path):

    try:
        # Create a W&B artifact
        artifact = wandb.Artifact(name="best_valloss_model_checkpoint", type="model")

        # Add the .pth file to the artifact
        artifact.add_file(ckpt_path)

        # Log the artifact using the existing run_logger
        run_logger.log_artifact(artifact)
    
    except Exception as e:
        print(f"Could not log artifact because of this error: {e}")
    
    return 

def train_model(model,
                train_loader,
                optimizer,
                scheduler,
                loss_function,
                device,
                val_model = False,
                val_loader = None,
                topN = False,
                topN_tup = None
                ):
    
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
                outputs = model(pixel_values = image_batch, train_mode = True)
        
        # perform regular inference on CPU
        else:
            outputs = model(pixel_values = image_batch, train_mode = True)


        optimizer.zero_grad()
        loss = loss_function(outputs, label_batch)
        batch_losses.append(loss.item())
        
        loss.backward()
        optimizer.step()

    train_losses = sum(batch_losses)/len(batch_losses)
    scheduler.step()

    if val_model:
        val_loss, val_acc = validate_model(model,
                                           val_loader,
                                           loss_function,
                                           device,
                                           topN,
                                           topN_tup)
        return (train_losses, val_loss, val_acc)

    else: return train_losses


def setup_training(data_paths,
                   num_epochs,
                   model,
                   device,
                   batch_size = 512,
                   patience=16,
                   min_delta_loss=1e-8,
                   min_epochs=20,
                   smooth_k=7,
                   run_logger = None,
                   local_testing=False,
                   log_model = False,
                   topN = False,
                   topN_tup = None
                   ):
    
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
            train_loss, val_loss, val_acc = train_model(model = model,
                                                        train_loader = train_loader,
                                                        optimizer = optimizer,
                                                        scheduler=scheduler,
                                                        loss_function=criterion,
                                                        device = device,
                                                        val_model=True,
                                                        val_loader=val_loader,
                                                        topN = topN,
                                                        topN_tup = topN_tup
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
            train_loss = train_model(model = model,
                                     train_loader =train_loader,
                                     optimizer = optimizer,
                                     scheduler = scheduler,
                                     loss_function=criterion,
                                     device = device,
                                     val_model=False,
                                     topN = topN,
                                     topN_tup = topN_tup
                                     )
            # Log only in remote experiments, not local
            if not local_testing: run_logger.log({"train_loss": train_loss})

    checkpoint = torch.load(ckpt_path, map_location=device)
    print(f"Best model in terms of validation loss was at: {checkpoint['epoch']+1} epoch, loading it for testing.")
    model.load_state_dict(checkpoint['model_state'])
    test_loss, test_acc = test_model(model,
                                     test_loader,
                                     criterion,
                                     device,
                                     topN,
                                     topN_tup)
    
    if log_model:
        log_model_to_wandb(run_logger, ckpt_path)

    if not local_testing:
        run_logger.log({"test_accuracy": test_acc, "test_loss": test_loss}); run_logger.finish()
    

    return (test_loss, test_acc)


def get_model(model_name, model_config):
    
    if model_name == 'radar_softdegrade':
        model = ViTRADAR_SoftDegrade(
            distance_metric=model_config['distance_metric'],
            aggregate_method=model_config['aggregate_method'],
            seq_select_method = model_config['seq_select_method'],
            num_out_classes = model_config['num_out'],
            aggregate_dim = model_config['aggregate_dim'],
            norm_type = model_config['norm_type'],
            return_anchors = model_config['return_anchors'],
            perc_ape = model_config['perc_ape'],
            corrupt_imp_weights=model_config['corrupt_imp_weights']
        )

    if model_name == 'radar_softanchor_v1':
        model = ViTRADAR_SoftAnchor_v1(
            distance_metric=model_config['distance_metric'],
            aggregate_method=model_config['aggregate_method'],
            seq_select_method = model_config['seq_select_method'],
            num_out_classes = model_config['num_out'],
            add_coordinates = model_config['add_coordinates'],
            K = model_config['K'],
            aggregate_dim = model_config['aggregate_dim'],
            norm_type = model_config['norm_type'],
            return_anchors = model_config['return_anchors'],
            perc_ape=model_config['perc_ape'],
            corrupt_imp_weights=model_config['corrupt_imp_weights']
                                        )
    
    if model_name == 'single_peg_cpvt':
        model = ViTWithPEG(
            num_labels=model_config['num_out'],
            perc_ape = model_config['perc_ape'],
            k = model_config['k']
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

    # This is project name in yaml config file, not the model name in get_model
    yaml_project_name = "single_peg_cpvt"; log_metrics = True; log_model = True
    
    configs_path = "./configs_train.yaml"
    data_paths = {"train_data": "./data/cifar10_train_cachegpu/train.pt",
                  "val_data" : "./data/cifar10_train_cachegpu/val.pt",
                  "test_data" : "./data/cifar10_train_cachegpu/test.pt" 
                  }
    
    config_details = get_project_details(configs_path, yaml_project_name)
    set_system_seed(config_details['config']['system_seed'])
    
    model = get_model(model_name = config_details['model_name'],
                      model_config = config_details['config']
                      )
    
    device = get_device()

    if log_metrics:
        run_logger = init_wandb(team_name=config_details['team_name'],
                    project_name=config_details['project_name'],
                    run_name = config_details['run_name'] + '_' + str(config_details['config']['system_seed'] ),
                    secret_key=config_details['secret_key'],
                    additional_config = config_details['config']
                    )
        
        run_logger.log_code("./")

        # Create a Complete model profile: Trainable Params, FLOPS etc including custom ops
        model_profile_dict = profile_models(
                                                model = model,
                                                example_input = torch.rand((1,3,224,224)),
                                                total_tr_rows = 35000, # This may need to change for CIFAR100
                                                batch_size = 512 if torch.cuda.is_available() else 64,
                                                num_epochs = config_details['config']['num_epochs']
                                            )
        
        model_profile_dict['dataset']  = 'cifar100'
        run_logger.log({"model_profile": model_profile_dict})

    test_loss, test_acc = setup_training(data_paths = data_paths,
                   num_epochs = config_details['config']['num_epochs'],
                   model = model,
                   device=device,
                   batch_size = 512 if torch.cuda.is_available() else 64,
                   patience=10,
                   min_delta_loss=1e-8,
                   min_epochs=20,
                   smooth_k=3,
                   run_logger = run_logger if log_metrics else None,
                   local_testing= not log_metrics,
                   log_model = log_model,
                   topN = True,
                   topN_tup = (1,5)
                   )
    
    print("Test Loss: ", test_loss)
    print("Test accuracy: ", test_acc)
