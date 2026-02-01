import torch
import random
import yaml
import numpy as np
import wandb
import torch.distributed as dist
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    Dataset,
    random_split
)

from torchvision.transforms import v2 as transforms
from torchvision import datasets
from typing import Optional, Tuple
import os
from helpers_profiling import *
from typing import Tuple, Optional, Dict, Any

IMAGENET_MEAN = (0.485, 0.456, 0.406); IMAGENET_STD  = (0.229, 0.224, 0.225)

class OxfordPetSeg(Dataset):
    def __init__(self, data_dir, split="train"):
        base_split = "trainval" if split != "test" else "test"
        self.base = datasets.OxfordIIITPet(
            root=data_dir,
            split=base_split,
            target_types="segmentation",
            download=True,
            transform=None,
            target_transform=None,
        )

        n = len(self.base)
        self.indices = range(0, n)
        if split == "train":
             # read full train and val combined data set and split later
            self.common_tf = transforms.Compose([
                transforms.ToImage(),
                transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                transforms.RandomHorizontalFlip(),
            ])
        else:
            # read full test data set and split later
            self.common_tf = transforms.Compose([
                transforms.ToImage(),
                transforms.Resize((224, 224)),
            ])

        self.img_tf = transforms.Compose([
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img, mask = self.base[self.indices[idx]]
        
        # if trainval - crop/flip/resize; if test - toimage, resize
        img, mask = self.common_tf(img, mask)

        # Same image transforms to trian and test images
        img = self.img_tf(img)
        
        # Same mask transforms to train and test images
        mask = mask.squeeze(0).to(torch.long)
        
        return img, mask

# Early stopping class for single node training
class EarlyStopping_SA:
    """
    Simple EarlyStopping for plain PyTorch training loops.
    Saves the best checkpoint and stops after `patience` non-improving epochs.
    """

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = "min",
                 ckpt_path: str = "best_by_metric.pth"):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.ckpt_path = ckpt_path

        self.best = float("inf") if mode == "min" else -float("inf")
        self.num_bad_epochs = 0

    def _is_improvement(self, metric_value: float) -> bool:
        if self.mode == "min":
            return metric_value < (self.best - self.min_delta)
        return metric_value > (self.best + self.min_delta)

    def step(self,
             metric_value: float,
             *,
             epoch: int,
             model,
             optimizer=None,
             extra: Optional[Dict[str, Any]] = None) -> bool:
        metric_value = float(metric_value)
        if self._is_improvement(metric_value):
            self.best = metric_value
            self.num_bad_epochs = 0

            payload: Dict[str, Any] = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metric": metric_value,
            }
            if optimizer is not None:
                payload["optimizer_state"] = optimizer.state_dict()
            if extra:
                payload.update(extra)

            torch.save(payload, self.ckpt_path)
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience

# Early stopping class for multi node training
class EarlyStopping_MW:
    """
    Simple EarlyStopping for plain PyTorch training loops.
    Saves the best checkpoint and stops after `patience` non-improving epochs.
    """

    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = "min",
                 ckpt_path: str = "best_by_metric.pth"):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.mode = mode
        self.ckpt_path = ckpt_path

        self.best = float("inf") if mode == "min" else -float("inf")
        self.num_bad_epochs = 0

    def _is_improvement(self, metric_value: float) -> bool:
        if self.mode == "min":
            return metric_value < (self.best - self.min_delta)
        return metric_value > (self.best + self.min_delta)

    def step(self,
             metric_value: float,
             *,
             epoch: int,
             model,
             optimizer=None,
             extra: Optional[Dict[str, Any]] = None) -> bool:
        metric_value = float(metric_value)
        if self._is_improvement(metric_value):
            self.best = metric_value
            self.num_bad_epochs = 0

            payload: Dict[str, Any] = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "metric": metric_value,
            }
            if optimizer is not None:
                payload["optimizer_state"] = optimizer.state_dict()
            if extra:
                payload.update(extra)

            torch.save(payload, self.ckpt_path)
            return False

        self.num_bad_epochs += 1
        return self.num_bad_epochs >= self.patience


def get_cifar10_loaders_optimized(data_dir: str='./data') -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:

    """Function that loads CIFAR10 data set from torchvision and applies transforms including resize to 224x224, normalization and augmentations.
        Normalization uses mean and std computed from CIFAR-10 dataset.
    """

    # CIFAR-10 mean and std for normalization. computed from training set.
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    # Training transforms with augmentation and resize to 224x224
    # TODO: Use Deterministic Horizontal Flip from paper, and may be extend to Deterministic Affine Transform as well.
    train_transform = transforms.Compose([
                            transforms.ToImage(),                                       # PIL -> tensor (uint8)
                            transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomAffine(degrees=0, translate=(0.1,0.1), interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.ToDtype(torch.float32, scale=True),              # now [0,1]
                            transforms.Normalize(mean, std),
                                        ])

    # Test transforms with resize to 224x224
    test_transform = transforms.Compose([
        transforms.ToImage(), # PIL -> Tensor fast path
        transforms.Resize((224, 224), interpolation = transforms.InterpolationMode.BILINEAR),  # Resize to ViT's expected input size
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std),
    ])

    # Load datasets
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset

def get_cifar100_loaders_optimized(data_dir: str='./data') -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Function that loads CIFAR100 data set from torchvision and applies transforms including resize to 224x224, normalization and augmentations.
        Normalization uses mean and std computed from CIFAR-100 dataset.
    """

    # CIFAR-100 mean and std for normalization. computed from training set.
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)

    # Training transforms with augmentation and resize to 224x224
    train_transform = transforms.Compose([
                            transforms.ToImage(),                                       # PIL -> tensor (uint8)
                            transforms.Resize((224,224), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomAffine(degrees=0, translate=(0.1,0.1), interpolation=transforms.InterpolationMode.BILINEAR),
                            transforms.ToDtype(torch.float32, scale=True),              # now [0,1]
                            transforms.Normalize(mean, std),
                                        ])

    # Test transforms with resize to 224x224
    test_transform = transforms.Compose([
        transforms.ToImage(), # PIL -> Tensor fast path
        transforms.Resize((224, 224), interpolation = transforms.InterpolationMode.BILINEAR),  # Resize to ViT's expected input size
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize(mean, std),
    ])

    # Load datasets
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=train_transform)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transform)

    return train_dataset, test_dataset

def get_oxfordseg_loaders(data_dir = './data'):

    train_dataset = OxfordPetSeg(data_dir=data_dir, split="train")
    test_dataset = OxfordPetSeg(data_dir=data_dir, split="test")

    return train_dataset, test_dataset
    
def save_cached_split(ds, path: str, batch_size: int=512, num_workers: int=8, dtype=torch.float16) -> None:

    """Saves tensor data in specified path with the raw torch data set provided."""
    
    dl = DataLoader(ds,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                    persistent_workers=True)
    Xs, Ys = [], []
    with torch.no_grad():
        for xb, yb in dl:
            Xs.append(xb.to(dtype).contiguous().cpu())
            Ys.append(yb.cpu())
    
    X = torch.cat(Xs); Y = torch.cat(Ys)
    torch.save({"images": X, "labels": Y}, path)


def split_pt_file_stratified(src_path,
                             split1_path="validation_ablations.pt",
                             split2_path="test_ablations.pt",
                             split1_frac=0.7,
                             seed=108,
                             split_stratified = True,
                             save_both_splits = True
                             ) -> None:
    
    """ Splits a single tensor file into two tensor files and writes them to disk as tensors.
    It assumes that single tensor file contains features and labels with anmes "images", "labels".

    Useful for splitting validation set of original experiment into val and test for ablations only.
    """

    d = torch.load(src_path, map_location= "cpu")
    # CIFAR-X , Oxford supports this
    X, Y = d["images"], d["labels"]
    assert X.shape[0] == Y.shape[0], "Mismatched X/Y lengths"
    g = torch.Generator().manual_seed(seed)

    split1_idx, split2_idx = [], []

    # Assumes labels are multi-class
    if split_stratified:

        # Stratified, balanced sampling
        for c in torch.unique(Y).tolist():
            
            idx = torch.where(Y == c)[0]
            perm = idx[torch.randperm(idx.numel(), generator=g)]

            n_val = max(1, min(idx.numel() - 1, int(idx.numel() * split1_frac)))
            
            split1_idx.append(perm[:n_val])
            split2_idx.append(perm[n_val:])
    
    # Labels are not assumed
    # standard random split of X, Y
    else:
        total_indices = torch.arange(X.shape[0])
        perm = total_indices[torch.randperm(total_indices.numel(), generator=g)]
        n_split1 = int(len(perm) * split1_frac)
        
        # Though lists are not required for splits. keeping them to avoid multiple downstream changes.
        split1_idx.append( perm[:n_split1] )
        split2_idx.append( perm[n_split1:] )

    split1_idx = torch.cat(split1_idx)
    split2_idx = torch.cat(split2_idx)

    # optional: shuffle within splits
    split1_idx = split1_idx[torch.randperm(split1_idx.numel(), generator=g)]
    split2_idx = split2_idx[torch.randperm(split2_idx.numel(), generator=g)]

    # CIFAR-X , Oxford supports this
    split1 = {"images": X[split1_idx].contiguous(), "labels": Y[split1_idx].contiguous()}
    split1_dataset = TensorDataset(split1["images"], split1["labels"])
    

    device_dtype = torch.float32
    if torch.cuda.is_available():
        device_dtype = torch.float16

    save_cached_split(split1_dataset, split1_path, dtype=device_dtype)
    print("Written Split 1 dataset ")
    

    if save_both_splits:
        # CIFAR-X , Oxford supports this
        split2 = {"images": X[split2_idx].contiguous(), "labels": Y[split2_idx].contiguous()}
        split2_dataset = TensorDataset(split2["images"], split2["labels"])
        
        save_cached_split(split2_dataset,   split2_path, dtype = device_dtype)
        print("Written Split 2 dataset ")

def make_cached_loader(path: str, batch_size: int=512, shuffle: bool =True, num_workers:int =8) -> DataLoader:

    """Function that loads pre-cached tensor data from specified path and returns a dataloader."""

    if not os.path.exists(path):
        raise ValueError(f"Cached data not found in path: {path}. Please run prepare_cached_datasets function first.")
    
    blob = torch.load(path, map_location="cpu")
    # CIFAR-X, Oxford supports this
    ds = TensorDataset(blob["images"], blob["labels"])
    
    pin_memory = False; persistent_workers = False
    if torch.cuda.is_available():
        pin_memory = True
        persistent_workers = True

    return DataLoader(ds,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=pin_memory,
                      persistent_workers=persistent_workers)


def prepare_cached_datasets(cached_data_path: str, dataset = "cifar10") -> dict:
    """ Function that prepares tensors from data set, splits into train and val and stores them in current project space.
    Uses torch.float16 if cuda is available, else torch.float32 on saved tensors.
    Cached path should end with / .

    Args:
        cached_data_path: Required -> Path to store cached tensors. Should end with / .
    Returns:
        data_paths: Dictionary containing paths to train, val and test tensor data.
    """
    
    # Check if pre-computed tensors already exist in specified folder.
    if os.path.exists(cached_data_path + "train.pt") and os.path.exists(cached_data_path + "val.pt") and os.path.exists(cached_data_path + "test.pt"):
        print("Cached data already exists. Skipping caching step.")
        data_paths = {'train_data': cached_data_path + "train.pt", 'val_data': cached_data_path + "val.pt", 'test_data': cached_data_path + "test.pt"}
        return data_paths
    
    # Get images data which are transformed with affines, resized to 224x224
    if dataset == "cifar10":
        train_raw , test_raw = get_cifar10_loaders_optimized(data_dir='./data')
    elif dataset == "cifar100":
        train_raw , test_raw = get_cifar100_loaders_optimized(data_dir='./data')
    elif dataset == "semseg_oxford":
        train_raw, test_raw  = get_oxfordseg_loaders(data_dir='./data')

    idx = torch.randperm(len(train_raw))
    cut = int(0.7 * len(train_raw))

    # Split train to train and validation sets
    train_ds = torch.utils.data.Subset(train_raw, idx[:cut])
    val_ds   = torch.utils.data.Subset(train_raw, idx[cut:])

    device_dtype = torch.float32
    if torch.cuda.is_available():
        device_dtype = torch.float16

    
    # Save pre-transformed image features into tensors
    os.makedirs(cached_data_path, exist_ok=True)

    save_cached_split(train_ds, cached_data_path + "train.pt", dtype=device_dtype)
    print("Written Train dataset ")
    save_cached_split(val_ds,   cached_data_path + "val.pt", dtype = device_dtype)
    print("written validation dataset")
    save_cached_split(test_raw, cached_data_path + "test.pt", dtype=device_dtype)
    print("written test dataset")

    data_paths = {'train_data': cached_data_path + "train.pt", 'val_data': cached_data_path + "val.pt", 'test_data': cached_data_path + "test.pt"}
    return data_paths


def profile_models(model, example_input, total_tr_rows, batch_size, num_epochs):

    """
    Function is a helper to compute FLOPs and Trainable parameters for overall and by-module of a model.
    """
    
    imp_modules = ['classifier', 'custom_pos_encoding', 'projection_phi', 'peg']
    imp_flop_metrics = ['forward_total_per_sample', 'forward_trainable_per_sample','train_per_sample',
                        'forward_per_step','train_per_step','train_per_epoch',
                        'train_full', 'num_train_samples','batch_size',
                        'num_epochs']

    profile_metrics = {}

    all_ops_model = get_all_inline_ops(model, example_input)
    dict_flops_model = flops_breakdown(model, example_input, all_ops_model, total_tr_rows, batch_size, num_epochs)
    flops_by_mod = dict_flops_model['by_module']

    # Extract FLOPS by module from flops breakdown
    for module_name in imp_modules:
        if module_name in flops_by_mod:
            profile_metrics[module_name] = flops_by_mod[module_name]

    
    # Extract other metrics from flops breakdown
    for module_name in imp_flop_metrics:
        if module_name in dict_flops_model:
            profile_metrics[module_name] = dict_flops_model[module_name]
    
    # compute and extract trainable other metrics from model
    # int, float, dict
    total_tr_params, total_tr_params_mb, tr_params_by_mod = count_parameters(model)
    
    profile_metrics['total_trainable_params'] = total_tr_params
    profile_metrics['total_trainable_params_mb'] = total_tr_params_mb
    profile_metrics['trainable_params_by_mod'] = tr_params_by_mod

    return profile_metrics


def set_system_seed(seed_num):
    os.environ["PYTHONHASHSEED"] = str(seed_num)
    torch.manual_seed(seed_num)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_num)
    random.seed(seed_num)
    np.random.seed(seed_num)

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

def get_val_splits(train_dataset, tr_size, val_size, tr_bs, val_bs):

    tr_dt, val_dt = random_split(train_dataset, [tr_size, val_size])

    train_loader = DataLoader(tr_dt, batch_size = tr_bs)
    val_loader = DataLoader(val_dt, batch_size = val_bs)

    return(train_loader, val_loader)

def init_wandb(team_name: str, project_name: str, run_name:str, secret_key:str, additional_config: dict = None):

    """" Function that initializes a WanDB session with the provided parameters."""

    try:
        wandb.login(key = secret_key, force = True)
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

def get_project_details(yaml_config_file, exp_name):

    with open(yaml_config_file, 'r') as file:
        loaded_config = yaml.safe_load(file)
    
    # loaded_config - 'yaml_project_name' , config, model_name, team_name, project_name, run_name, secret_key
    if exp_name in loaded_config:
        return loaded_config[exp_name]
    else:
        raise Exception("Provided experiment doesn't exist")

def _get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))

def _ddp_init_if_needed_v2(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    acc = torch.accelerator.current_accelerator()
    backend = torch.distributed.get_default_backend_for_device(acc)
    # initialize process group.
    dist.init_process_group(backend, rank = rank, world_size = world_size)

def _ddp_broadcast_stop(should_stop: bool, device: torch.device) -> bool:
    stop_tensor = torch.tensor([1 if should_stop else 0], device=device, dtype=torch.int32)
    dist.broadcast(stop_tensor, src=0)
    return bool(stop_tensor.item())

