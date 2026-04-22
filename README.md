# Sales RL Agent

This repository contains a script-driven implementation of a risk-sensitive distributional actor-critic for sales dialogue control, together with the paper, synthetic simulator, benchmark tooling, and inference utilities.

The project is intentionally separated from the presentation repo. The nested `bantr-presentation/` directory is ignored here and remains its own GitHub repository.

## Repository layout

- `paper.tex` and `paper.pdf`: research paper
- `sales_rl_core.py`: simulator, models, training loops, evaluation, plotting, checkpoint helpers
- `train_sales_rl_agent.py`: train scalar or distributional controllers and save checkpoints
- `use_sales_rl_agent.py`: load a checkpoint, score a manual state, or run simulator test rollouts
- `run_sales_benchmark.py`: reproduce the benchmark figures used in the paper
- `generate_architecture_figure.py`: regenerate the external architecture diagram used in the paper
- `sample_state.json`: example state input for the inference script
- `figures/`: paper figures
- `artifacts/`: benchmark summaries and metrics

## Environment setup

For local CPU work:

```bash
python -m pip install torch
python -m pip install -r requirements.txt
```

For an NVIDIA A100:

```bash
python -m pip install --upgrade pip
python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
python -m pip install -r requirements.txt
```

## Train the agent

Train the distributional agent on GPU and save a checkpoint:

```bash
python train_sales_rl_agent.py \
  --algorithm distributional_a2c \
  --device cuda \
  --batch-envs 256 \
  --hidden-dim 256 \
  --total-updates 480 \
  --evaluate-episodes 512
```

Train both scalar and distributional baselines and regenerate comparison figures:

```bash
python train_sales_rl_agent.py \
  --algorithm both \
  --device cuda \
  --batch-envs 256 \
  --hidden-dim 256 \
  --total-updates 480 \
  --evaluate-episodes 512
```

## Use the trained agent

Score a single manually specified sales state:

```bash
python use_sales_rl_agent.py \
  --checkpoint checkpoints/distributional_a2c.pt \
  --device cuda \
  --state-file sample_state.json
```

Run greedy simulator tests with the saved checkpoint:

```bash
python use_sales_rl_agent.py \
  --checkpoint checkpoints/distributional_a2c.pt \
  --device cuda \
  --simulate-episodes 5
```

## Reproduce paper assets

```bash
python generate_architecture_figure.py
python run_sales_benchmark.py
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
pdflatex -interaction=nonstopmode -halt-on-error paper.tex
```
