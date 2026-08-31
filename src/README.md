# src

- `main.py`: Hydra entrypoint. `ddp_setup()` -> build objects (`utils.get_train_objs`) -> `Trainer.train()`.
- `trainer.py`: the DDP training loop. snapshot/resume, metrics CSV+plot. Model-agnostic, shouldn't need edits.
- `utils.py`: `get_train_objs` wires up whatever the active `config/experiment/*.yaml` points at (model components, optimizer/scheduler, loss, dataset, collate_fn) via `hydra.utils.instantiate`.
- `models/`: `component_1` -> `component_2` -> `component_3` (`model.py`'s fixed 3-stage forward pass) + `loss/`. The `example.*` files are a minimal encoder -> latent -> decoder stand-in. Replace their contents (and the config that points at them, see `config/component/`) with your own architecture.
- `datasets/`: `mock.py` is a synthetic, dependency-free dataset so the whole stack runs with no real data. `collate/collate_function.py` turns raw dataset items into model-ready batches; swap it alongside your own dataset if the item format changes.
- `train_multi_node.sh`: PBS/torchrun launcher for a real HPC cluster. Cluster-specific things are marked `EDIT ME`.
