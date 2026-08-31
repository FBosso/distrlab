import csv
import os
import time

import matplotlib

import matplotlib.pyplot as plt

import torch
from utils import TrainerConfig, Snapshot, get_device
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import fsspec
from dataclasses import asdict

class Trainer():

    def __init__(
            self,
            trainer_config: TrainerConfig,
            model,
            optimizer,
            loss_fn,
            train_dataset,
            val_dataset=None,
            collate_fn=None,
            scheduler=None,
    ):
        self.config = trainer_config
        self.collate_fn = collate_fn

        # set torchrun vars
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])

        # device and backend agnostic accelleration (CUDA, MPS, XPU,...)
        self.device = get_device(self.local_rank)
        self.device_type = self.device.type

        # data
        self.train_dataset = train_dataset
        self.train_loader = self._prepare_dataloader(train_dataset)
        self.val_loader = self._prepare_dataloader(val_dataset) if val_dataset else None

        # init train state
        self.epochs_run = 0
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_every = self.config.save_every
        self.loss_fn = loss_fn

        # snapshot to resume training
        if self.config.snapshot_path is None:
            self.config.snapshot_path = "snapshot.pt"
        # if present, load it
        self._load_snapshot()
        
        self.metrics_history = self._load_metrics_history()

        # best val_loss seen so far (see train()'s best_snapshot.pt save below).
        # recovered from history on resume, so a resumed run doesn't reset to +inf and
        # overwrite a legitimately better best_snapshot.pt with a worse one.
        self.best_val_loss = min(
            (row["val_loss"] for row in self.metrics_history if row["val_loss"] is not None),
            default=float("inf"),
        )

        # let process wait till all are ready to run
        dist.barrier()
        # wrap model with DDP. device_ids must be None for CPU, [index] for an
        # accelerator device
        device_ids = None if self.device.type == "cpu" else [self.local_rank]
        self.model = DDP(self.model, device_ids=device_ids)

    def _prepare_dataloader(self, dataset: Dataset):
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            pin_memory=True,  # optimize the host-to-device transfer time
            shuffle=False,  # needs to be false when using distributedsampler
            num_workers=self.config.data_loader_workers,
            sampler=DistributedSampler(dataset),
            collate_fn=self.collate_fn,
        )

    def _load_snapshot(self):
        try:
            snapshot = fsspec.open(self.config.snapshot_path)
            with snapshot as f:
                snapshot_data = torch.load(f, map_location="cpu")
        except:
            print("Snapshot not found. Starting training from scratch")
            return

        # extract needed items from snapshot
        snapshot = Snapshot(**snapshot_data)
        self.model.load_state_dict(snapshot.model_state)
        self.optimizer.load_state_dict(snapshot.optimizer_state)
        self.epochs_run = snapshot.finished_epoch
        if self.scheduler is not None and snapshot.scheduler_state is not None:
            self.scheduler.load_state_dict(snapshot.scheduler_state)

        # print log
        print(f"Resuming training from snapshot at epoch {self.epochs_run}")

    def _metrics_paths(self):
        metrics_dir = os.path.dirname(self.config.snapshot_path) or "."
        return f"{metrics_dir}/metrics.csv", f"{metrics_dir}/metrics.png"

    def _load_metrics_history(self):
        csv_path, _ = self._metrics_paths()
        try:
            with fsspec.open(csv_path, "r", newline="") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            return []

        def to_float(value):
            return float(value) if value not in (None, "") else None

        return [
            {
                "epoch": int(row["epoch"]),
                "train_loss": to_float(row.get("train_loss")),
                "val_loss": to_float(row.get("val_loss")),
                "lr": to_float(row.get("lr")),
            }
            for row in rows
        ]

    def _save_metrics(self):
        csv_path, png_path = self._metrics_paths()

        fieldnames = ["epoch", "train_loss", "val_loss", "lr"]
        with fsspec.open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.metrics_history)

        epochs = [row["epoch"] for row in self.metrics_history]
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(epochs, [row["train_loss"] for row in self.metrics_history], label="train_loss")
        if any(row["val_loss"] is not None for row in self.metrics_history):
            ax1.plot(epochs, [row["val_loss"] for row in self.metrics_history], label="val_loss")
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss")
        ax1.set_title("Training metrics")
        ax1.legend(loc="upper left")

        if any(row["lr"] is not None for row in self.metrics_history):
            ax2 = ax1.twinx()
            ax2.plot(epochs, [row["lr"] for row in self.metrics_history], label="lr", linestyle="--", color="gray")
            ax2.set_ylabel("learning rate")
            ax2.legend(loc="upper right")

        fig.tight_layout()
        with fsspec.open(png_path, "wb") as f:
            fig.savefig(f, format="png")
        plt.close(fig)

        print(f"Metrics saved to {csv_path} and {png_path}")

    def _move_to_device(self, obj):
        # recursively move any nesting of tensors (list/tuple/dict) to self.device,
        # leaving non-tensor entries (e.g. None for an absent modality) untouched
        if torch.is_tensor(obj):
            return obj.to(self.device, non_blocking=True)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self._move_to_device(item) for item in obj)
        if isinstance(obj, dict):
            return {key: self._move_to_device(value) for key, value in obj.items()}
        return obj

    def _run_batch(self, x, y, train: bool = True):
        with torch.set_grad_enabled(train):
            pred = self.model(x)
            loss = self.loss_fn(pred, y)

        if train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def _run_epoch(self, epoch: int, dataloader: DataLoader, train: bool = True):
        # if False is passed, is like doing model.eval()
        self.model.train(train)

        # DistributedSampler reuses the same shuffle order every epoch unless told
        # which epoch it is. only the train loader needs this (val is shuffle=False anyway).
        if isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)

        total_loss = 0.0
        for x, y in dataloader:
            x = self._move_to_device(x)
            y = self._move_to_device(y)
            loss = self._run_batch(x, y, train=train)
            total_loss += loss

        local_avg = total_loss / len(dataloader)
        # average across GPUs so the reported/logged/plotted loss reflects the whole batch
        # across all ranks, not just this rank's local shard.
        loss_tensor = torch.tensor(local_avg, device=self.device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        return (loss_tensor / dist.get_world_size()).item()

    def _best_snapshot_path(self):
        return os.path.join(os.path.dirname(self.config.snapshot_path) or ".", "best_snapshot.pt")

    def _save_snapshot(self, epoch, path=None):
        path = path or self.config.snapshot_path
        model = self.model

        # for DDP training this should always be true, just to be safe
        raw_model = model.module if hasattr(model, "module") else model
        snapshot = Snapshot(
            model_state=raw_model.state_dict(),
            optimizer_state=self.optimizer.state_dict(),
            finished_epoch=epoch,
            scheduler_state=self.scheduler.state_dict() if self.scheduler is not None else None,
        )
        # save the snapshot
        snapshot = asdict(snapshot)
        torch.save(snapshot, path)

        #print log
        label = "Best snapshot" if path != self.config.snapshot_path else "Snapshot"
        print(f"{label} saved at epoch {epoch}")

    def _log_epoch(self, epoch, train_loss, val_loss, lr, elapsed):
        self.metrics_history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "lr": lr}
        )
        val_str = f"{val_loss:>10.4f}" if val_loss is not None else f"{'--':>10}"
        lr_str = f"{lr:>10.6f}" if lr is not None else f"{'--':>10}"
        print(
            f"{epoch:>10d} | {train_loss:>10.4f} | {val_str} | {lr_str} | {elapsed:>6.1f}s"
        )

    def train(self):
        if self.global_rank == 0:
            print(f"{'epoch':>10} | {'train_loss':>10} | {'val_loss':>10} | {'lr':>10} | {'time':>7}")
            print("-" * 63)

        for epoch in range(self.epochs_run, self.config.max_epochs):
            epoch += 1  # from 0-based to 1-based
            start_time = time.time()

            train_loss = self._run_epoch(epoch, self.train_loader, train=True)
            # step the scheduler before snapshotting, so a resumed run's scheduler state
            # is never behind the persisted finished_epoch.
            if self.scheduler is not None:
                self.scheduler.step()
            if self.global_rank == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)
            # eval run
            val_loss = self._run_epoch(epoch, self.val_loader, train=False) if self.val_loader else None

            if self.global_rank == 0:
                lr = self.scheduler.get_last_lr()[0] if self.scheduler is not None else None
                self._log_epoch(epoch, train_loss, val_loss, lr, time.time() - start_time)
                if val_loss is not None and val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_snapshot(epoch, path=self._best_snapshot_path())
                if epoch % self.save_every == 0:
                    self._save_metrics()

        if self.global_rank == 0:
            self._save_metrics()
            print("Training complete.")
