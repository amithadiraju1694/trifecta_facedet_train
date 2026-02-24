import os
import sys
import torch
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from distributed_trainer import Trainer
from types import SimpleNamespace
import hashlib
import inspect
from helpers import (
    profile_models,
    get_project_details,
    set_system_seed,
    WiderFaceTrain,
    collate_widerface,
                    )


from Custom_VIT_SemSeg import (
    ViTRADAR_SoftAnchor_v1_SemSeg,
    ViTLoRA_SemSeg,
    ViTWithSegFormer,
    ViTFaceDetectorPlain,
    ViTFaceDetectorRADAR
                                )



def get_model_semseg(model_name, model_config):
    model_config = _cfg_to_dict(model_config)
    model = None

    if model_name == 'radar_softanchor_v1':
        model = ViTRADAR_SoftAnchor_v1_SemSeg(
            distance_metric=model_config['distance_metric'],
            aggregate_method=model_config['aggregate_method'],
            seq_select_method = model_config['seq_select_method'],
            
            num_out_classes = model_config['num_out'],
            transpose_convolutions = model_config['transpose_convolutions'],
            add_coordinates = model_config['add_coordinates'],

            K = model_config['K'],
            aggregate_dim = model_config['aggregate_dim'],
            norm_type = model_config['norm_type'],

            return_anchors = model_config['return_anchors'],
            perc_ape=model_config['perc_ape'],
            corrupt_imp_weights=model_config['corrupt_imp_weights']
                                        )
    
    
    if model_name == "vit_lora":
        model = ViTLoRA_SemSeg(
                 num_out_classes = model_config['num_out'],
                 transpose_convolutions = model_config['transpose_convolutions'],

                 r = model_config['r'],
                 lora_alpha = model_config['lora_alpha'],
                 lora_dropout = 0.05,
                 target_module = model_config["target_module"]
                            )
    
    if model_name == "vit_segform":

        model = ViTWithSegFormer(
            num_out_classes=model_config['num_out'],
            transpose_convolutions = model_config['transpose_convolutions'],
        )

    if model is None:
        raise ValueError(f"Unsupported semseg model_name: {model_name}")
    return model


def _cfg_to_dict(model_config):
    if isinstance(model_config, dict):
        return model_config
    if isinstance(model_config, SimpleNamespace):
        return vars(model_config)
    try:
        return dict(model_config)
    except Exception:
        return vars(model_config)


def _build_model_from_signature(model_cls, model_config):
    cfg = _cfg_to_dict(model_config)
    sig = inspect.signature(model_cls.__init__)
    kwargs = {}
    for key in sig.parameters:
        if key == "self":
            continue
        if key in cfg:
            kwargs[key] = cfg[key]
    return model_cls(**kwargs)


def get_model_facedet(exp_name, config):
    exp_key = str(exp_name).lower().strip()
    model_config = getattr(config, "model_config", config)

    if exp_key == "vit_radar_fd" or ("radar" in exp_key and exp_key.endswith("_fd")):
        return _build_model_from_signature(ViTFaceDetectorRADAR, model_config)

    if exp_key == "vit_plain_fd" or ("plain" in exp_key and exp_key.endswith("_fd")):
        return _build_model_from_signature(ViTFaceDetectorPlain, model_config)

    # Backward-compatible fallback if header naming is not yet updated.
    train_name = str(getattr(getattr(config, "train_config", None), "model_name", "")).lower()
    if train_name in {"facedet_radar", "vitfacedet_radar", "vit_face_detector_radar", "radar_softanchor_v1"}:
        return _build_model_from_signature(ViTFaceDetectorRADAR, model_config)
    if train_name in {"facedet_plain", "vitfacedet_plain", "vit_face_detector_plain"}:
        return _build_model_from_signature(ViTFaceDetectorPlain, model_config)

    raise ValueError(
        f"Unsupported facedet experiment header: {exp_name}. "
        "Use headers like 'vit_radar_fd' or 'vit_plain_fd'."
    )

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus

def ddp_setup():
    acc = torch.accelerator.current_accelerator()
    rank = int(os.environ["LOCAL_RANK"] )
    device: torch.device = torch.device(f"{acc}:{rank}")
    backend = torch.distributed.get_default_backend_for_device(device)
    init_process_group(backend=backend)
    torch.accelerator.set_device_index(rank)

# Can be removed after testing
def _as_namespace(obj):
    if isinstance(obj, SimpleNamespace):
        return obj
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _as_namespace(v) for k, v in obj.items()})
    if isinstance(obj, (list, tuple)):
        return type(obj)(_as_namespace(v) for v in obj)
    return obj


def _stable_fraction(key: str, seed: int) -> float:
    h = hashlib.blake2b(digest_size=8)
    h.update(str(seed).encode("utf-8"))
    h.update(b"/")
    h.update(key.encode("utf-8"))
    return int.from_bytes(h.digest(), "big") / float(1 << 64)


def _expand_wds_urls(urls):
    if urls is None:
        return []
    if isinstance(urls, (list, tuple)):
        return list(urls)
    if isinstance(urls, str):
        s = urls.strip()
        if not s:
            return []
        try:
            import webdataset as wds

            return list(wds.shardlists.expand_urls(s))
        except Exception:
            return [u.strip() for u in s.split(",") if u.strip()]
    raise TypeError(f"Unsupported WebDataset urls type: {type(urls)}")


def _split_shards_for_val(urls, val_fraction: float, seed: int):
    expanded = _expand_wds_urls(urls)
    if not expanded:
        return [], []
    val_urls = [u for u in expanded if _stable_fraction(u, seed) < val_fraction]
    if not val_urls:
        val_urls = expanded[: max(1, int(round(len(expanded) * val_fraction)))]
    val_set = set(val_urls)
    train_urls = [u for u in expanded if u not in val_set]
    if not train_urls:
        train_urls = expanded
    return train_urls, val_urls

def get_urls_from_hf(dataset_hf_url:str, split_val:bool, val_fraction: float = 0.3, seed:int = 108):

    from huggingface_hub import HfFileSystem, hf_hub_url
    
    token = "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"
    
    hf_base = dataset_hf_url.strip()
    if not hf_base.startswith("hf://"):
        hf_base = "hf://datasets/" + hf_base.lstrip("/")
    hf_base = hf_base.rstrip("/") + "/"

    splits = {"train": "**/*-train-*.tar", "validation": "**/*-validation-*.tar"}
    fs = HfFileSystem()
    train_files = [fs.resolve_path(path) for path in fs.glob(hf_base + splits["train"])]
    test_files = [fs.resolve_path(path) for path in fs.glob(hf_base + splits["validation"])]

    train_urls = [hf_hub_url(f.repo_id, f.path_in_repo, repo_type="dataset") for f in train_files]
    test_urls = [hf_hub_url(f.repo_id, f.path_in_repo, repo_type="dataset") for f in test_files]

    if not train_urls:
        raise ValueError(f"No train shards found under {dataset_hf_url!r} with pattern {splits['train']!r}.")
    if not test_urls:
        raise ValueError(f"No validation/test shards found under {dataset_hf_url!r} with pattern {splits['validation']!r}.")
    
    if split_val:
        train_urls, val_urls = _split_shards_for_val(train_urls, val_fraction=val_fraction, seed=seed)
    else:
        val_urls = []

    curl_prefix = "pipe:curl -s -L"
    if token:
        curl_prefix = f"{curl_prefix} -H 'Authorization:Bearer {token}'"

    train_shards = [f"{curl_prefix} {u}" for u in train_urls]
    val_shards = [f"{curl_prefix} {u}" for u in val_urls] if split_val else None
    test_shards = [f"{curl_prefix} {u}" for u in test_urls]

    return train_shards, test_shards, val_shards

# This may be redundant given CLS usecase is removed
def build_streaming_wds_cls_loaders(
    train_shards,
    test_shards,
    batch_size: int,
    num_workers: int,
    *,
    val_shards=None,
    image_size: int = 224,
    shuffle_buf: int = 10_000,
    prefetch_factor: int = 2,
):
    import webdataset as wds
    from torchvision import transforms as T

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)

    train_transform = T.Compose(
        [
            T.RandomResizedCrop(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )
    eval_transform = T.Compose(
        [
            T.Resize(int(image_size * 256 / 224), interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    train_urls = _expand_wds_urls(train_shards)
    val_urls = _expand_wds_urls(val_shards) if val_shards != None else []
    if val_urls:
        val_set = set(val_urls)
        train_urls = [u for u in train_urls if u not in val_set] or train_urls

    test_urls = _expand_wds_urls(test_shards)
    if not test_urls:
        raise ValueError("Missing test WebDataset shard urls. Set train_config.test_shards or env WDS_TEST_SHARDS.")

    def _make_ds(urls, *, is_train: bool):
        if not urls:
            return None

        ds = wds.WebDataset(
            urls,
            shardshuffle=is_train,
            handler=wds.ignore_and_continue,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        )
        if is_train and shuffle_buf > 0:
            ds = ds.shuffle(shuffle_buf)

        img_key = "jpg;jpeg;png;webp"
        label_key = "cls;class;label"
        ds = (
            ds.decode("pil")
            .to_tuple(img_key, label_key)
            .map_tuple(train_transform if is_train else eval_transform, int)
            .batched(batch_size, partial=False)
        )
        return ds

    train_ds = _make_ds(train_urls, is_train=True)
    val_ds = _make_ds(val_urls, is_train=False) if val_shards != None else None
    test_ds = _make_ds(test_urls, is_train=False)

    def _make_loader(ds):
        if ds is None:
            return None
        loader_kwargs = {
            "batch_size": None,
            "num_workers": num_workers,
            "pin_memory": True,
            "persistent_workers": (num_workers > 0),
        }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = prefetch_factor
        return wds.WebLoader(ds, **loader_kwargs)

    train_loader = _make_loader(train_ds)
    val_loader = _make_loader(val_ds) if val_shards != None else None
    test_loader = _make_loader(test_ds)
    return train_loader, val_loader, test_loader


def build_widerface_ddp_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int,
    image_size: int,
    download: bool = True,
):
    dist_init = dist.is_available() and dist.is_initialized()
    rank = dist.get_rank() if dist_init else 0

    if dist_init and download:
        if rank == 0:
            # Download once to avoid multiple ranks writing same files.
            _ = WiderFaceTrain(root=data_root, split="train", image_size=image_size, download=True)
            _ = WiderFaceTrain(root=data_root, split="val", image_size=image_size, download=True)
        dist.barrier()
        download = False

    train_ds = WiderFaceTrain(root=data_root, split="train", image_size=image_size, download=download)
    val_ds = WiderFaceTrain(root=data_root, split="val", image_size=image_size, download=download)

    train_sampler = None
    val_sampler = None
    if dist_init:
        world_size = dist.get_world_size()
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)

    loader_common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": (num_workers > 0),
        "collate_fn": collate_widerface,
    }

    train_loader = DataLoader(
        train_ds,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        **loader_common,
    )
    val_loader = DataLoader(
        val_ds,
        shuffle=False,
        sampler=val_sampler,
        **loader_common,
    )
    # WIDER FACE test split does not provide boxes; use val split as test/eval.
    test_loader = val_loader
    return train_loader, val_loader, test_loader

def main(yaml_config_file: str, exp_name: str):
    ddp_setup()

    config = _as_namespace(get_project_details(yaml_config_file, exp_name))
    seed = int(getattr(config.train_config, "system_seed", 108))
    rank = int(os.environ.get("RANK", "0"))
    set_system_seed(seed + rank)

    task = str(config.train_config.task).lower()

    if task == "semseg":
        get_model = get_model_semseg
    elif task == "facedet":
        get_model = get_model_facedet
    else:
        raise ValueError(f"Unsupported task: {config.train_config.task}")

    if task == "facedet":
        model = get_model(exp_name=exp_name, config=config)
    else:
        model = get_model(model_name=config.train_config.model_name,
                          model_config=config.model_config
                          )
    
    num_workers = int(getattr(config.train_config, "data_loader_workers", "8"))
    split_val = bool(getattr(config.train_config, "split_val", True))
    val_fraction = float(getattr(config.train_config, "val_fraction", 0.05))

    image_size = int(getattr(config.train_config, "image_size", 224))
    shuffle_buf = int(getattr(config.train_config, "shuffle_buf", 10_000))

    if task == "facedet":
        data_root = getattr(config.train_config, 
                            "dataset_root", None) or getattr(config.train_config,
                                                              "data_root", None)
        if not data_root:
            raise ValueError("Missing dataset_root/data_root for facedet task.")
        
        dataset_download = bool(getattr(config.train_config, "dataset_download", True))

        train_dataloader, val_dataloader, test_dataloader = build_widerface_ddp_loaders(
            data_root=data_root,
            batch_size=int(config.train_config.batch_size),
            num_workers=num_workers,
            image_size=image_size,
            download=dataset_download,
        )
    
    # Given that classification is removed from this repo , entire else condition may be redundant 
    # can be removed.
    else:

        dataset_hf_url = getattr(config.train_config, "dataset_hf_url", None)
        if not dataset_hf_url:
            raise ValueError("Missing dataset_hf_url. Set train_config.dataset_hf_url or env WDS_DATASET_HF_URL.")

        train_shards, test_shards, val_shards = get_urls_from_hf(dataset_hf_url = dataset_hf_url,
                                                                 split_val = split_val,
                                                                 val_fraction = val_fraction,
                                                                 seed=seed)
        
        train_dataloader, val_dataloader, test_dataloader = build_streaming_wds_cls_loaders(
            train_shards=train_shards,
            val_shards=val_shards,
            test_shards=test_shards,
            batch_size=int(config.train_config.batch_size),
            num_workers=num_workers,
            image_size=image_size,
            shuffle_buf=shuffle_buf,
        )
        if not split_val:
            val_dataloader = None

    
    trainer = Trainer(
                    config=config,
                    model=model,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    test_dataloader=test_dataloader,
                               )
    
    # Trains, optionally validates, logs model and metrics
    trainer.train()

    # Runs eval on test/val set and logs metrics
    trainer.test()

    destroy_process_group()

if __name__ == "__main__":
    _min_gpu_count = 2
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):
        print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
        sys.exit()

    main("./configs_train.yaml", "vit_radar_fd")
