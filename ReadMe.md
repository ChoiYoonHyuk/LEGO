
## Overview

<img width="1045" height="501" alt="Image" src="https://github.com/user-attachments/assets/bd7fd4d0-dcbe-40ce-b061-8a88becbf72e" />

This repository provides a PyTorch Geometric implementation of a "Shifted Graph-Coupled Activation with Learned Edge Targets for GNNs."


## Datasets

The `main.py` script supports the following datasets indexed from `0` to `8`:

- `0`: Roman-empire
- `1`: Minesweeper
- `2`: Amazon-ratings
- `3`: Chameleon
- `4`: Squirrel
- `5`: Actor
- `6`: Cornell
- `7`: Texas
- `8`: Wisconsin

## Requirements

Install the main dependencies before running the code:

```bash
pip install torch numpy
pip install torch-geometric
```

Depending on your environment, you may also need the matching PyTorch Geometric extension packages for your PyTorch and CUDA setup.

## Execution

To train the model on the supported benchmark datasets:

```bash
python main.py 0   # Roman-empire
python main.py 1   # Minesweeper
python main.py 2   # Amazon-ratings
python main.py 3   # Chameleon
python main.py 4   # Squirrel
python main.py 5   # Actor
python main.py 6   # Cornell
python main.py 7   # Texas
python main.py 8   # Wisconsin
```

## Useful Arguments

You can also configure training and model hyperparameters from the command line:

```bash
python main.py 3 --device auto --hidden_dim 128 --num_layers 2 --dropout 0.6
```

Key options include:
- `--device`: `auto`, `cuda`, `mps`, `cpu`
- `--seed`: random seed
- `--split_idx`: split index for datasets with multiple predefined splits
- `--hidden_dim`: hidden feature dimension
- `--edge_hidden_dim`: hidden dimension for edge networks
- `--num_layers`: number of Shifted ProxAct layers
- `--num_relations`: number of learned relation prototypes
- `--num_basis`: number of basis components in the bounded influence potential
- `--alpha`, `--kappa`, `--num_pd_iter`, `--num_newton`, `--xi`, `--mu_max`: proximal optimization parameters
- `--dropout`, `--lr`, `--weight_decay`, `--weight_reg`, `--offset_reg`
- `--label_smoothing`, `--grad_clip`, `--max_epochs`, `--patience`, `--log_every`

## Output

During training, the script reports:
- training loss,
- validation accuracy,
- test accuracy,
- best validation/test scores,
- mean learned edge weight,
- mean offset norm,
- proximal gate value,
- and GPU memory usage when CUDA is available.

