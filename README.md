# Distribute Homework 1

This repository contains multi-system machine learning comparison experiments implemented in Python.

## Included Experiments

- `multi_system_compare.py`: classic non-vision baseline (single machine / dask distributed / pytorch MLP)
- `vision_multi_system_compare.py`: visual task (CIFAR-10) comparison with softmax and ResNet18 transfer baseline
- `resnet_cross_system_compare.py`: **ResNet18-only** comparison across:
  - single-machine system
  - distributed system (PyTorch DDP local simulation, configurable for remote nodes)
  - deep learning system (full fine-tuning)

## Result Files

- `outputs/` : baseline results
- `outputs_vision/` : vision comparison results
- `outputs_resnet/` : ResNet18 cross-system comparison results

Each output folder includes:
- `comparison_results.csv`
- `comparison_results.json`
- `comparison_results.md`
- configuration snapshots (`installation_config.json`, and when available `experiment_config.json`)

## Notes

- Datasets are downloaded at runtime and are **not** tracked in this repository.
- Model binaries are excluded via `.gitignore` to keep repository size manageable.
