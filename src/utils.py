from dataclasses import dataclass
from typing import Optional, Any, Dict
from collections import OrderedDict
import torch
from torch.distributed import init_process_group
import os
from models.model import Model
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch.utils.data import random_split

#dataclasses
@dataclass
class TrainerConfig:
    max_epochs: int = None
    batch_size: int = None
    data_loader_workers: int = None
    snapshot_path: Optional[str] = None
    save_every: int = None

@dataclass
class Snapshot:
    model_state: "OrderedDict[str, torch.Tensor]"
    optimizer_state: Dict[str, Any]
    finished_epoch: int
    scheduler_state: Optional[Dict[str, Any]] = None


#main imported functions
def get_device(rank):
    acc = torch.accelerator.current_accelerator(check_available=True)
    return torch.device(f"{acc.type}:{rank}") if acc is not None else torch.device("cpu")

def ddp_setup():
    rank = int(os.environ["LOCAL_RANK"])

    #code for device and backend agnostic execution
    device = get_device(rank)
    backend = torch.distributed.get_default_backend_for_device(device)
    # init_process_group's device_id must be an accelerator device with an index. CPU
    # takes none
    if device.type == "cpu":
        init_process_group(backend=backend)
    else:
        torch.accelerator.set_device_index(rank)
        init_process_group(backend=backend, device_id=device)

def get_train_objs(cfg):

    #components
    component_1 = instantiate(cfg.component_1)
    component_2 = instantiate(cfg.component_2)
    component_3 = instantiate(cfg.component_3)

    #model, optimizer (+ its optional scheduler), loss
    model = Model(component_1, component_2, component_3)

    optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
    scheduler_cfg = optimizer_cfg.pop("scheduler", None)
    optimizer = instantiate(optimizer_cfg, model.parameters())
    scheduler = instantiate(scheduler_cfg, optimizer) if scheduler_cfg is not None else None

    loss_fn = instantiate(cfg.loss)

    #dataset
    dataset = instantiate(cfg.dataset)
    if hasattr(dataset, "train_val_split"):
        # a dataset can define this to control its own train/val split (e.g. splitting
        # by some grouping key so a given group's samples never end up on both sides)
        train_set, val_set = dataset.train_val_split(cfg.dataset.train_split, seed=42)
    else:
        train_len = int(len(dataset) * cfg.dataset.train_split)
        train_set, val_set = random_split(
            dataset, [train_len, len(dataset) - train_len],
            generator=torch.Generator().manual_seed(42)
        )

    # collate_fn does dataset-format -> model-format conversion (see
    # src/datasets/collate/collate_function.py). None if not needed
    # config/experiment/*.yaml controls whether cfg.collate exists.
    collate_fn = instantiate(cfg.collate) if cfg.get("collate") is not None else None

    return model, optimizer, scheduler, loss_fn, train_set, val_set, collate_fn
