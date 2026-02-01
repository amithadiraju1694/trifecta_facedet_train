from dataclasses import dataclass, asdict
from collections import OrderedDict
from typing import Any, Dict
import os
import fsspec

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics import JaccardIndex
from torchmetrics.classification import Accuracy
from helpers import init_wandb, log_model_to_wandb, set_system_seed


@dataclass
class Snapshot:
    model_state: 'OrderedDict[str, torch.Tensor]'
    optimizer_state: Dict[str, Any]
    finished_epoch: int

class Trainer:
    def __init__(self,
                 config,
                 model,
                 train_dataloader,
                 val_dataloader = None,
                 test_dataloader = None
                 ):
        
        self.config = config
        
        # set torchrun variables
        self.local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.global_rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))

        self.base_seed = int(getattr(self.config.train_config, "system_seed", 0))
        self.seed = self.base_seed + self.global_rank
        set_system_seed(self.seed)

        self.deterministic = bool(getattr(self.config.train_config, "deterministic", True))
        self.loss_fn = nn.CrossEntropyLoss()

        # basic accel flags setting deterministic cuda flags
        if self.deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
            if torch.cuda.is_available():
                torch.set_float32_matmul_precision("highest")
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:
                torch.use_deterministic_algorithms(True)
        elif torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
            torch.backends.cudnn.benchmark = True

        # only include trainable params to optimizer. Need to rebuild or add a new param group if adding new layers later
        params = [p for p in model.parameters() if p.requires_grad]
        scaled_lr = self.config.model_config.lr * self.world_size
        self.optimizer = torch.optim.AdamW(params,
                                  lr=scaled_lr,
                                  weight_decay = self.config.model_config.weight_decay
                                  )
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer,
                                                                     T_0=10,
                                                                     T_mult=1,
                                                                     eta_min=1e-6)

        
        # set device
        self.acc = torch.accelerator.current_accelerator()
        self.device: torch.device = torch.device(f"{self.acc}:{self.local_rank}")
        self.device_type = self.device.type

        self.train_loader = train_dataloader
        self.val_loader = val_dataloader if val_dataloader else None
        self.test_loader = test_dataloader if test_dataloader else None
        
        # initialize train states
        self.epochs_run = 0
        self.model = model.to(self.device)
        self.save_every = self.config.train_config.save_every
        self.eval_every = getattr(self.config.train_config, "eval_every", 3)

        if self.config.train_config.task == "class":
            self.metric_var = Accuracy(task = "multiclass",
                                       num_classes=self.config.model_config.num_out
                                       ).to(self.device)
        elif self.config.train_config.task == "semseg":
            self.metric_var = JaccardIndex(task="multiclass",
                                           num_classes=self.config.model_config.num_out
                                           ).to(self.device)
        else:
            raise ValueError(f"Task: {self.task} is not supported.")

        if self.config.train_config.use_amp:
            self.scaler = torch.amp.GradScaler(self.device_type)

        self.patience = self.config.train_config.patience
        self.min_delta = 1e-8
        self.best_val_loss = float("inf")
        # To be used only when val data set is not provided.
        self.best_train_loss = float("inf")
        self.epochs_no_improve = 0

        self.run_logger = None
        if self.global_rank == 0 and getattr(self.config.wandb_config, "log_metrics", False):
            self.run_logger = init_wandb(team_name=self.config.wandb_config.team_name,
                                         project_name=self.config.wandb_config.project_name,
                                         run_name=self.config.wandb_config.run_name,
                                         secret_key=self.config.wandb_config.secret_key,
                                         additional_config=getattr(self.config.wandb_config, "__dict__", None))
        
        self._load_snapshot()
        
        # wrap with DDP. this step will synch model across all the processes.
        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank, broadcast_buffers=False)

    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.config.train_config.snapshot_path, "rb")
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")
        except FileNotFoundError:
            print("Snapshot not found. Training model from scratch")
            return 

        snapshot = Snapshot(**snapshot_data)
        target = self.model.module if hasattr(self.model, "module") else self.model
        target.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets, train: bool = True,compute_metrics: bool = False) -> float:
        
        """Runs a single batch of data using AMP and gradient scaling"""
        
        self.model.train() if train else self.model.eval()
        with torch.set_grad_enabled(train), torch.amp.autocast(device_type=self.device_type,dtype=torch.bfloat16,enabled=(self.config.train_config.use_amp)):
            logits = self.model(source)
            loss = self.loss_fn(logits, targets)
        
        if train:
            # if mixed precision is to be used
            self.optimizer.zero_grad(set_to_none=True)
            if self.config.train_config.use_amp: 
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)

                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config.train_config.grad_norm_clip
                                               )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            
            # else normal backward step with gradient clipping
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.config.train_config.grad_norm_clip)
                self.optimizer.step()
        
        if not compute_metrics:
            return loss.item()
        else:
            return(loss.item(), self.metric_var(logits, targets).item())

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True, compute_metrics: bool = False):
        
        if train and hasattr(dataloader, "sampler") and hasattr(dataloader.sampler, "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        loss_sum = 0.0
        metric_sum = 0.0
        n_batches = 0
            
        for iter, (source, targets) in enumerate(dataloader):
            
            step_type = "Train" if train else "Eval"
            source = source.to(self.device,
                               non_blocking = True,
                               memory_format=torch.channels_last
                               )
            targets = targets.to(self.device, non_blocking = True)
            
            # run this single batch of data using DDP
            batch_out = self._run_batch(source, targets, train, compute_metrics)
            if compute_metrics:
                batch_loss, batch_metric = batch_out
                metric_sum += float(batch_metric)
            else:
                batch_loss = batch_out
            loss_sum += float(batch_loss)
            n_batches += 1
            
            if self.global_rank == 0 and iter % 100 == 0:
                print(f"[RANK{self.global_rank }] Epoch {epoch} | Iter {iter
                             } | {step_type} Loss {batch_loss:.5f}"
                     )
        
        # Compute avg loss of all batches in this machine
        # and poll from all other machines
        avg_loss = loss_sum / max(1, n_batches)
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_loss = (t / self.world_size).item()

        if not compute_metrics:
            return avg_loss

        # Compute avg metric of all batches in this machine
        # and poll from all other machines
        avg_metric = metric_sum / max(1, n_batches)
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor(avg_metric, device=self.device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            avg_metric = (t / self.world_size).item()

        return (avg_loss, avg_metric)

    def _save_snapshot(self, epoch):
        # capture snapshot
        model = self.model
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch
        )
        # save snapshot
        snapshot = asdict(snapshot)
        with fsspec.open(self.config.train_config.snapshot_path, "wb") as f:
            torch.save(snapshot, f)
            
        print(f"Snapshot saved at epoch {epoch}")

    def train(self):
        
        for epoch in range(self.epochs_run, self.config.train_config.max_epochs):
            epoch += 1
            # run one epoch on entire data set
            train_loss = self._run_epoch(epoch, self.train_loader, train=True)
            self.lr_scheduler.step(epoch)

            # save recent model snapshot
            if self.global_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            
            should_stop = False
            
            # eval run
            if self.val_loader != None and epoch % self.eval_every == 0:
                # Run validation data set
                val_loss, val_metric = self._run_epoch(epoch,
                                         self.val_loader,
                                         train=False,
                                         compute_metrics=True
                                         )

                # On main machine, based on val loss check if it needs to be stopped
                # or found best model so far, checkpoint if so
                if self.global_rank == 0:
                    if val_loss < (self.best_val_loss - self.min_delta):
                        self.best_val_loss = val_loss
                        self.epochs_no_improve = 0
                        self._save_snapshot(epoch)
                    else:
                        self.epochs_no_improve += 1
                        should_stop = self.epochs_no_improve >= self.patience

                # Communicate to other machine if it needs to be stopped
                if dist.is_available() and dist.is_initialized():
                    stop_tensor = torch.tensor([1 if should_stop else 0], device=self.device, dtype=torch.int32)
                    dist.broadcast(stop_tensor, src=0)
                    should_stop = bool(stop_tensor.item())

                # Log to WanDB from main node.
                if self.global_rank == 0 and self.run_logger is not None:
                    self.run_logger.log({"train_loss": train_loss, "val_loss": val_loss, "val_metric": val_metric})
            
            # Logn and optionally check early stopping on train loss
            else:
                if self.global_rank == 0 and self.run_logger is not None:
                    self.run_logger.log({"train_loss": train_loss})

                # If val loader is not provided at all
                # decide whether to early stop only based on train loss
                if self.val_loader is None:

                    # decide whether to stop or if best model is found
                    if self.global_rank == 0:
                        if train_loss < (self.best_train_loss - self.min_delta):
                            self.best_train_loss = train_loss
                            self.epochs_no_improve = 0
                        else:
                            self.epochs_no_improve += 1
                            should_stop = self.epochs_no_improve >= self.patience
                    
                    # communicate to all machines based on stop_iteration accordingly
                    if dist.is_available() and dist.is_initialized():
                        stop_tensor = torch.tensor([1 if should_stop else 0], device=self.device, dtype=torch.int32)
                        dist.broadcast(stop_tensor, src=0)
                        should_stop = bool(stop_tensor.item())

            # Check if need to be early stopped according to machine rank
            if should_stop:
                if self.global_rank == 0:
                    print(f"Early stopping at epoch {epoch}.")
                break
    
    def test(self):
        # Load latest model snapshot for inference
        self._load_snapshot()

        if not self.test_loader:
            raise ValueError("Test Data loader not provided during intialization")

        test_loss, test_metric = self._run_epoch(1,
                                self.test_loader,
                                train = False,
                                compute_metrics = True
                                        )
        
        if self.global_rank == 0 and self.run_logger is not None:
            self.run_logger.log({"test_loss": test_loss, "test_metric": test_metric})
            log_model_to_wandb(self.run_logger, self.config.train_config.snapshot_path)
            self.run_logger.finish()
        
        

        
