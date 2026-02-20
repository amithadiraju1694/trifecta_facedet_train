import torch
import random
import yaml
import numpy as np
import wandb
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    Dataset
)

from torchvision.transforms import v2 as transforms
from torchvision import datasets
from typing import Optional, Tuple
import os
from helpers_profiling import *
from typing import Tuple, Optional, Dict, Any, List


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


class WiderFaceTrain(Dataset):
    """WIDER FACE map-style dataset wrapper with resized xyxy boxes."""

    def __init__(self, root: str, split: str = "train", image_size: int = 224, download: bool = True):
        if split not in ("train", "val"):
            raise ValueError("WiderFaceTrain supports only 'train' or 'val' splits.")

        self.image_size = int(image_size)
        self.split = split
        self.ds = datasets.WIDERFace(
            root=root,
            split=split,
            download=download,
            transform=None,
        )

        self.transform = transforms.Compose([
            transforms.ToImage(),
            transforms.Resize((self.image_size, self.image_size), antialias=True),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx: int):
        img, target_raw = self.ds[idx]
        orig_w, orig_h = img.size

        img_t = self.transform(img)

        boxes_xywh = target_raw.get("bbox", None)
        if boxes_xywh is None:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            return img_t, boxes

        if not torch.is_tensor(boxes_xywh):
            boxes_xywh = torch.tensor(boxes_xywh, dtype=torch.float32)
        boxes_xywh = boxes_xywh.float()

        if boxes_xywh.numel() == 0:
            boxes = torch.empty((0, 4), dtype=torch.float32)
            return img_t, boxes

        boxes_xywh = boxes_xywh.view(-1, 4)
        sx = self.image_size / float(orig_w)
        sy = self.image_size / float(orig_h)

        x = boxes_xywh[:, 0]
        y = boxes_xywh[:, 1]
        w = boxes_xywh[:, 2]
        h = boxes_xywh[:, 3]

        x1 = x * sx
        y1 = y * sy
        x2 = (x + w) * sx
        y2 = (y + h) * sy
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, self.image_size - 1)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, self.image_size - 1)

        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        boxes = boxes[valid]
        return img_t, boxes


def collate_widerface(batch: List[Tuple[torch.Tensor, torch.Tensor]]):
    images = torch.stack([item[0] for item in batch], dim=0)
    boxes = [item[1] for item in batch]
    return images, boxes


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

def cxcywh_to_xyxy(box_cxcywh: torch.Tensor) -> torch.Tensor:
    # box_cxcywh: (..., 4) in [cx,cy,w,h]
    cx, cy, w, h = box_cxcywh.unbind(-1)
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return torch.stack([x1, y1, x2, y2], dim=-1)

def nms_xyxy(boxes: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.4) -> torch.Tensor:
    # boxes: (N,4), scores: (N,)
    # returns indices kept
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    idxs = torch.argsort(scores, descending=True)
    keep = []

    while idxs.numel() > 0:
        i = idxs[0]
        keep.append(i)

        if idxs.numel() == 1:
            break

        rest = idxs[1:]
        iou = box_iou_xyxy(boxes[i].unsqueeze(0), boxes[rest]).squeeze(0)
        idxs = rest[iou <= iou_thresh]

    return torch.stack(keep, dim=0)

def box_iou_xyxy(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # boxes1: (N,4), boxes2: (M,4)
    # returns IoU (N,M)
    N = boxes1.shape[0]
    M = boxes2.shape[0]
    if N == 0 or M == 0:
        return torch.zeros((N, M), device=boxes1.device)

    x11, y11, x12, y12 = boxes1[:, 0], boxes1[:, 1], boxes1[:, 2], boxes1[:, 3]
    x21, y21, x22, y22 = boxes2[:, 0], boxes2[:, 1], boxes2[:, 2], boxes2[:, 3]

    # intersection
    inter_x1 = torch.max(x11[:, None], x21[None, :])
    inter_y1 = torch.max(y11[:, None], y21[None, :])
    inter_x2 = torch.min(x12[:, None], x22[None, :])
    inter_y2 = torch.min(y12[:, None], y22[None, :])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    area1 = (x12 - x11).clamp(min=0) * (y12 - y11).clamp(min=0)
    area2 = (x22 - x21).clamp(min=0) * (y22 - y21).clamp(min=0)

    union = area1[:, None] + area2[None, :] - inter_area
    iou = inter_area / union.clamp(min=1e-6)
    return iou


def build_grid_targets(
    targets: List[Any],
    grid: int,
    image_size: int,
    device: torch.device,
):
    bsz = len(targets)
    obj_tgt = torch.zeros((bsz, 1, grid, grid), dtype=torch.float32, device=device)
    box_tgt = torch.zeros((bsz, 4, grid, grid), dtype=torch.float32, device=device)
    box_msk = torch.zeros((bsz, 1, grid, grid), dtype=torch.float32, device=device)

    for b in range(bsz):
        boxes = extract_boxes_xyxy(targets[b]).to(device=device, dtype=torch.float32)
        if boxes.numel() == 0:
            continue

        boxes_norm = boxes.clone()
        boxes_norm[:, 0::2] = boxes_norm[:, 0::2] / float(image_size)
        boxes_norm[:, 1::2] = boxes_norm[:, 1::2] / float(image_size)
        boxes_cxcywh = xyxy_to_cxcywh(boxes_norm)

        for n in range(boxes_cxcywh.shape[0]):
            cx, cy, w, h = boxes_cxcywh[n]
            gx = int(torch.clamp(cx * grid, 0, grid - 1).item())
            gy = int(torch.clamp(cy * grid, 0, grid - 1).item())

            if obj_tgt[b, 0, gy, gx] > 0:
                prev_w = box_tgt[b, 2, gy, gx]
                prev_h = box_tgt[b, 3, gy, gx]
                if (w * h) <= (prev_w * prev_h):
                    continue

            obj_tgt[b, 0, gy, gx] = 1.0
            box_tgt[b, :, gy, gx] = torch.tensor([cx, cy, w, h], device=device)
            box_msk[b, 0, gy, gx] = 1.0

    return obj_tgt, box_tgt, box_msk

def xyxy_to_cxcywh(box_xyxy: torch.Tensor) -> torch.Tensor:
    # box_xyxy: (..., 4) in [x1,y1,x2,y2]
    x1, y1, x2, y2 = box_xyxy.unbind(-1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    return torch.stack([cx, cy, w, h], dim=-1)


def extract_boxes_xyxy(target_item: Any) -> torch.Tensor:
    if isinstance(target_item, dict):
        if "boxes_xyxy" not in target_item:
            raise ValueError("Face-detector targets dict must contain 'boxes_xyxy'.")
        box_tensor = target_item["boxes_xyxy"]
    else:
        box_tensor = target_item

    if not torch.is_tensor(box_tensor):
        box_tensor = torch.tensor(box_tensor, dtype=torch.float32)
    box_tensor = box_tensor.float().view(-1, 4)
    return box_tensor

