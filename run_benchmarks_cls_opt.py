import os
import sys
import torch
from torch.distributed import init_process_group, destroy_process_group
from distributed_trainer import Trainer
from types import SimpleNamespace
import hashlib
from helpers import (
    profile_models,
    get_project_details,
    set_system_seed
                    )

from Custom_VIT import (
    ViTRADAR_SoftDegrade,
    ViTRADAR_SoftAnchor_v1,
    ViTWithPEG,
    ViTWithStaticPositionalEncoding,
    ViTLoRAClassifier,
    ViTWithSetTransformerHead,
    ViTWithConvGPSAHead
                        )

from Custom_VIT_SemSeg import (
    ViTRADAR_SoftDegrade_SemSeg,
    ViTRADAR_SoftAnchor_v1_SemSeg,
    ViTWithPEG_SemSeg,
    ViTLoRA_SemSeg,
    ViTWithMask2FormerSeg,
    ViTWithSegFormer
                                )

def get_model_class(model_name, model_config):
    
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
    
    if model_name == "vit_lora":
        model = ViTLoRAClassifier(
                 num_out_classes = model_config['num_out'],
                 r = model_config['r'],
                 lora_alpha = model_config['lora_alpha'],
                 lora_dropout = 0.05,
                 target_module = model_config["target_module"]
                            )
    
    if model_name == "vit_settrans":

        model = ViTWithSetTransformerHead(
                 num_out_classes = model_config['num_out'],
                 m_inducing = model_config["m_inducing"],
                 n_heads = model_config["n_heads"]
                            )
    
    if model_name == "vit_convgpsa":

        model = ViTWithConvGPSAHead(
            num_out_classes=model_config['num_out'],
            convit_heads=model_config['convit_heads'],
            mlp_ratio=model_config['mlp_ratio']
        )

    return model

def get_model_semseg(model_name, model_config):
    
    if model_name == 'radar_softdegrade':
        model = ViTRADAR_SoftDegrade_SemSeg(
            distance_metric=model_config['distance_metric'],
            aggregate_method=model_config['aggregate_method'],
            seq_select_method = model_config['seq_select_method'],

            num_out_classes = model_config['num_out'],
            transpose_convolutions = model_config['transpose_convolutions'],
            
            aggregate_dim = model_config['aggregate_dim'],
            norm_type = model_config['norm_type'],
            return_anchors = model_config['return_anchors'],
            perc_ape = model_config['perc_ape'],
            corrupt_imp_weights=model_config['corrupt_imp_weights']
        )

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
    
    if model_name == 'single_peg_cpvt':
        model = ViTWithPEG_SemSeg(
            num_out_classes = model_config['num_out'],
            transpose_convolutions = model_config['transpose_convolutions'],

            perc_ape = model_config['perc_ape'],
            k = model_config['k']
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
    
    if model_name == "vit_mask2form":

        model = ViTWithMask2FormerSeg(
                 num_out_classes = model_config['num_out'],
                 transpose_convolutions = model_config['transpose_convolutions'],
                            )
    
    if model_name == "vit_segform":

        model = ViTWithSegFormer(
            num_out_classes=model_config['num_out'],
            transpose_convolutions = model_config['transpose_convolutions'],
        )

    return model

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    has_gpu = torch.accelerator.is_available()
    gpu_count = torch.accelerator.device_count()
    return has_gpu and gpu_count >= min_gpus

def ddp_setup():
    acc = torch.accelerator.current_accelerator()
    rank = int(os.environ["LOCAL_RANK"])
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

def main(yaml_project_name: str):
    ddp_setup()

    config = get_project_details(yaml_project_name)
    seed = int(getattr(config.train_config, "system_seed", 108))
    rank = int(os.environ.get("RANK", "0"))
    set_system_seed(seed + rank)

    get_model = get_model_class if config.train_config.task == "class" else get_model_semseg

    model = get_model(model_name=config.train_config.model_name,
                      model_config=config.model_config
                      )
    
    num_workers = int(getattr(config.train_config, "data_loader_workers", "8"))
    split_val = bool(getattr(config.train_config, "split_val", True))
    val_fraction = float(getattr(config.train_config, "val_fraction", 0.05))

    image_size = int(getattr(config.train_config, "image_size", 224))
    shuffle_buf = int(getattr(config.train_config, "shuffle_buf", 10_000))

    dataset_hf_url = getattr(config.train_config, "dataset_hf_url", None)
    if not dataset_hf_url:
        raise ValueError("Missing dataset_hf_url. Set train_config.dataset_hf_url or env WDS_DATASET_HF_URL.")

    train_shards, test_shards, val_shards = get_urls_from_hf(dataset_hf_url = dataset_hf_url,
                                                             split_val = split_val,
                                                             val_fraction = val_fraction,
                                                             seed=seed)
    
    loaders = build_streaming_wds_cls_loaders(
        train_shards=train_shards,
        val_shards=val_shards,
        test_shards=test_shards,

        batch_size=int(config.train_config.batch_size),
        num_workers=num_workers,
        split_val=split_val,

        val_fraction=val_fraction,
        seed=seed,
        image_size=image_size,

        shuffle_buf=shuffle_buf,
                                            )
    
    # use loaders returned from above function, split accordingly
    if split_val:
        train_dataloader, val_dataloader, test_dataloader = loaders
    
    else:
        train_dataloader, test_dataloader = loaders
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
    if len(sys.argv) < 3:
        print("Usage: torchrun ... run_benchmarks_cls_opt.py <configs_train.yaml>")
        sys.exit(2)
    main(sys.argv[1])
