# distrlab

**Distributed Training Lab (DisTrLab)** is an open-source, Hydra-based experimentation
template. Pull it and get straight to your experiments, with a clear path to scaling them
across hundreds of GPUs without wiring anything yourself.

You write the code for your architecture, drop it in the right place, and pair it with a
config file per component. That's it, you're ready to scale. Want to swap a backbone for
a benchmark, or go from "an idea on my laptop" to a DDP run on an HPC cluster? Thanks to
Hydra, that's just a config change: one `Trainer` (`src/trainer.py`), one config tree, 
and the same `src/main.py` drive every launch path, local or PBS/torchrun on HPC. 
Interrupted runs resume automatically too: model, optimizer, and scheduler state are 
checkpointed regularly by design.

Ships with a minimal, synthetic-data example (encoder -> latent -> decoder, trained on a
generated sine-mixture series) so the whole stack runs with zero setup. Swap it for your
own model and data. The plumbing around it doesn't change.

## Quickstart

```bash
uv sync
uv run torchrun --standalone --nproc_per_node=1 src/main.py
```

Trains the example model for a few epochs on synthetic data, writing a snapshot, a
resumable `metrics.csv`/`metrics.png`, and logs under `outputs/example/<timestamp>/`.

## Structure

- `config/`: Hydra config tree. `config.yaml` sets defaults + trainer params.
  `config/experiment/*.yaml` selects a full experiment (which model components, dataset,
  loss, optimizer); `config/component/*/` holds the swappable pieces each experiment
  picks from. See `config/experiment/example.yaml` for the pattern.
- `src/`: see `src/README.md`.

## Running it

**Locally, single node:**
```bash
uv run torchrun --standalone --nproc_per_node=<N> src/main.py [experiment=<name>] [hydra overrides...]
```

**On an HPC cluster (PBS):**
```bash
qsub -v experiment=<name>,hydra_overrides="trainer_config.max_epochs=25" src/train_multi_node.sh
```
Edit the lines marked `EDIT ME` (queue, resources, module names) for your cluster first.


## Adapting this to your own project

1. Write your `Dataset` under `src/datasets/` (see `mock.py` for the expected item
   shape) and a matching `config/component/dataset/*.yaml`.
2. Replace `src/models/component_1/2/3` with your own architecture (that can have an 
   arbitrary number and type of components) `model.py`'s forward pass is a fixed 
   `component_1 -> component_2 -> component_3` pipeline. Add more stages there if three 
   doesn't fit.
3. Make a new `config/experiment/*.yaml` referencing your components, and make it the default
   in `config/config.yaml`.
4. Everything else (`Trainer`, `main.py`, the launch scripts) should keep working
   unmodified.

## Citation

If this template is useful in your work, please cite it:

```bibtex
@software{distrlab,
  author = {FBosso},
  title = {distrlab: a Hydra-based experimentation template for local and HPC training},
  url = {https://github.com/FBosso/distrlab},
  year = {2026}
}
```

See [`CITATION.cff`](./CITATION.cff) for the machine-readable version.

## License

MIT — see [`LICENSE`](./LICENSE).
