# Custom_PE_ViT


### Pre-Computing Transformed Image Datasets

We strongly suggesting pre-computing all image transformations and saving them on disk ONCE, and re-use them in all ablations and training experiments, so as to cut down compute and cost of triaing and ablations.

Here's how you can do so:

#### 1. Create Train, Val and Test splits from original data set
```
exp_data_paths = prepare_cached_datasets(cached_data_path = "./data/cifarx_train_cachegpu") (OR)

exp_data_paths = prepare_cached_datasets(cached_data_path = "./data/cifarx_train_cachecpu")

```

#### 2. Split Train's Validation Set into further val and test splits, suitable for ablation studies, with same training code.

```
# Take Validation Tensors and split it to validation and test for ablations
split_pt_file_stratified(
    src_path = "./data/cifar10_train_cachegpu/val.pt",
    split1_path="./data/cifar10_ablations_cachegpu/val_ablations.pt",
    split2_path="./data/cifar10_ablations_cachegpu/test_ablations.pt",
    split1_frac=0.6,
    seed = 108,
    save_both_splits= True
)
```

#### 3. (OPTIONALLY) Reduce Training dataset of full experiment to half its size onyl for ablations

```
# Take Validation Tensors and split it to validation and test for ablations
split_pt_file_stratified(
    src_path = "./data/cifar10_train_cachegpu/train.pt",
    split1_path="./data/cifar10_ablations_cachegpu/train_ablations.pt",
    split2_path="",
    split1_frac=0.6,
    seed = 108,
    save_both_splits= False
)

```