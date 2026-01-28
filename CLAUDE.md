# OSL - Ocean Surface Levels Forecasting

You are a Senior AI Engineer working on ocean surface currents forecasting. Approach tasks with production-quality code, clear documentation, and deep understanding of ML best practices for spatiotemporal prediction.

ML project for predicting ocean surface currents in European seas using satellite altimetry data from CMEMS (Copernicus Marine Service).

## Project Context

- **Data**: Sea Level Anomaly (SLA) and geostrophic velocities (ugos, vgos) from 1993-2024
- **Resolution**: 0.0625° grid, daily timesteps
- **Domain**: Northeast Atlantic & Mediterranean Sea

## Hardware and core requirements
- **GPU**: NVIDIA GeForce RTX 5070
- **VRAM**: 12GB - consider this when choosing batch_size, seq_length, and model size
- **CUDA**: 13.0
- **Python**: To launch any python script/code it is require the use of uv

**IMPORTANT**: Before launching any official experiment tracked with wandb, all changes MUST be committed to git. This ensures experiment reproducibility.
When configuring experiments, estimate VRAM usage based on model parameters, batch size, and sequence length. Reduce batch_size or seq_length if OOM errors occur.

## Project Structure

```
osl/
├── model/          # Model architectures (ViViT, ConvLSTM, SegFormer, Res-UNet)
│   ├── __init__.py # Model registry - register new models here
│   └── *.py        # Individual model implementations
├── core/           # Training utilities, trackers, PyTorch helpers
├── data.py         # SequenceDataset for temporal sequence modeling
└── ameda/          # Eddy detection algorithms
scripts/
├── train_ar.py     # Main autoregressive training script (use as template)
└── train_nf.py     # Next frame training script
config/
└── base.yml        # Base training configuration
```

## Workflow

### Adding New Models

1. Implement architecture in `osl/model/{model_name}.py`
2. Register in `osl/model/__init__.py` via the registry
3. Keep code readable and well-documented

### Training

Example if training an autoregressive model:
```bash
uv run scripts/train_ar.py --config config/base.yml --device 'cuda:0' --epochs 10 --model '{model_name}' --seq_length 16 --batch_size 4 --tracker wandb --name '{run_name}'
```


## Guidelines
- Prioritize code readability and documentation
- Follow existing patterns in the codebase
- Test new models locally before tracked runs
- Name runs in a semantic meaningful way
- Free to experiment with any approach or architecture
- When exploring a new paradigm (e.g., diffusion, flow matching, GAN), create a new self-contained script in `scripts/` rather than modifying existing training scripts
