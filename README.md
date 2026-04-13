# Food-101 Scratch vs Fine-Tuning

This project compares training from scratch and fine-tuning on Food-101 with two classical CNN backbones: `ResNet18` and `DenseNet121`.

## Project Structure

```text
homework2/
├── configs/
│   └── experiments.yaml
├── data/
│   ├── processed/
│   └── raw/
├── dataset/
├── scripts/
│   ├── prepare_data.py
│   ├── run_ann.sh
│   └── run_experiments.py
├── src/
│   └── food101_exp/
├── README.md
├── requirements.txt
└── .gitignore
```

## Packages and Versions

The experiments were developed and run with the following package versions:

| Package | Version |
| --- | --- |
| Python | 3.11.15 |
| torch | 2.11.0 |
| torchvision | 0.26.0 |
| Pillow | 12.1.1 |
| PyYAML | 6.0.3 |
| pandas | 2.3.3 |
| matplotlib | 3.10.8 |

`requirements.txt` keeps the corresponding Python package versions used by the project.

## Dataset Placement

The repository keeps empty dataset directories only. Place the original Food-101 archive at:

```text
dataset/food-101.tar.gz
```

Then extract it with:

```bash
python scripts/prepare_data.py
```

After extraction, the dataset will be located under:

```text
data/raw/food-101/
```

## Experiments

The project includes four formal experiments:

- `resnet_scratch`
- `resnet_finetune`
- `densenet_scratch`
- `densenet_finetune`

Run all experiments:

```bash
bash scripts/run_ann.sh python scripts/run_experiments.py --config configs/experiments.yaml
```

Run a single experiment:

```bash
bash scripts/run_ann.sh python scripts/run_experiments.py --config configs/experiments.yaml --experiment resnet_scratch
```

Resume training from the latest checkpoint:

```bash
bash scripts/run_ann.sh python scripts/run_experiments.py --config configs/experiments.yaml --experiment resnet_scratch --resume last --epochs 50
```

## Main Training Settings

Shared settings:

- input size: `224`
- batch size: `48`
- optimizer: `AdamW`
- weight decay: `1e-4`
- label smoothing: `0.1`
- num workers: `8`
- AMP: enabled

Per-experiment settings:

- `resnet_scratch`: `epochs=50`, `lr=0.001`
- `resnet_finetune`: `epochs=25`, `head lr=0.001`, `backbone lr=0.0001`
- `densenet_scratch`: `epochs=30`, `lr=0.001`
- `densenet_finetune`: `epochs=15`, `head lr=0.001`, `backbone lr=0.0001`

## Outputs

- `checkpoints/<experiment>/best.pt`: best checkpoint
- `checkpoints/<experiment>/last.pt`: latest checkpoint for resume
- `logs/<experiment>.log`: epoch-level training log
- `results/<experiment>_history.csv`: training history
- `results/<experiment>_curves.png`: training curves
