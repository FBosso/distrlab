import hydra
from torch.distributed import destroy_process_group
from utils import ddp_setup
from omegaconf import DictConfig
from utils import TrainerConfig, get_train_objs
from trainer import Trainer




@hydra.main(
    version_base=None, config_path="../config", config_name="config"
)
def main(cfg: DictConfig):
    ddp_setup()
    try:
        trainer_cfg = TrainerConfig(**cfg.trainer_config)
        model, optimizer, scheduler, loss_fn, train_dataset, val_dataset, collate_fn = get_train_objs(cfg)
        trainer = Trainer(
            trainer_cfg, model, optimizer, loss_fn,
            train_dataset, val_dataset, collate_fn, scheduler=scheduler,
        )
        trainer.train()
    finally:
        destroy_process_group()

if __name__=="__main__":
    main()