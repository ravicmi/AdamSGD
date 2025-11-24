# Quick Start Guide

## Installation (5 minutes)

1. **Activate virtual environment** (already created):
```bash
cd AdamSGD
source adamVenv/bin/activate
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

This will install:
- PyTorch 2.0+ with MPS support
- torchvision for datasets
- matplotlib, seaborn for plotting
- pyyaml for configuration
- tqdm for progress bars

## Running Experiments

### Option 1: Run All Experiments at Once

```bash
python run_all_experiments.py
```

This will:
- Create all necessary directories
- Download MNIST and CIFAR-10 datasets automatically
- Run all 4 experiments sequentially
- Generate comparison plots
- Save results to `results/` directory

**Estimated time**: 
- Fast mode (default): 15-30 minutes
- Full mode: 2-6 hours

### Option 2: Run Individual Experiments

```bash
# Logistic Regression on MNIST (~3-5 min in fast mode)
python experiments/exp_6_1_logistic_regression.py

# MLP on MNIST (~5-10 min in fast mode)
python experiments/exp_6_2_mlp.py

# CNN on CIFAR-10 (~10-20 min in fast mode)
python experiments/exp_6_3_cnn.py

# Advanced experiments with ablations (~10-15 min in fast mode)
python experiments/exp_6_4_advanced.py
```

### Option 3: Run Selected Experiments

```bash
# Only run experiments 6.1 and 6.2
python run_all_experiments.py --experiments 6.1 6.2
```

## Configuration

Before running, you can edit `config/config.yaml`:

### Switch Between Fast and Full Mode

```yaml
# In config/config.yaml
mode: 'fast'  # Quick testing (default)
# OR
mode: 'full'  # Paper-matching experiments
```

### Adjust Epochs for Fast Mode

```yaml
fast:
  epochs:
    logistic_regression: 10  # Increase for better results
    mlp: 20
    cnn: 30
    advanced: 20
```

## Viewing Results

After running, check the `results/` directory:

```bash
# View directory structure
ls -R results/

# Results are organized by experiment:
# results/exp_6_1_logistic_regression/
#   - comparison_train_loss.png
#   - comparison_val_loss.png
#   - comparison_train_acc.png
#   - comparison_val_acc.png
#   - adam_curves.png
#   - sgd_curves.png
#   - etc.
```

Open the PNG files to view:
- Training/validation loss curves
- Training/validation accuracy curves
- Optimizer comparisons

## What to Expect

### Experiment 6.1 (Logistic Regression)
- **5 optimizers tested**: Adam, SGD, SGD+Momentum, AdaGrad, RMSProp
- **Expected**: Adam converges fastest, reaches ~92% test accuracy
- **Plots**: 5 individual curves + 4 comparison plots

### Experiment 6.2 (MLP)
- **4 optimizers tested**: Adam, SGD+Momentum, AdaGrad, RMSProp
- **Expected**: Adam reaches ~98% test accuracy, better than others
- **Plots**: 4 individual curves + 4 comparison plots

### Experiment 6.3 (CNN on CIFAR-10)
- **4 optimizers tested**: Adam, SGD+Momentum, AdaGrad, RMSProp
- **Expected**: Adam shows more stable convergence
- **Note**: CIFAR-10 is harder, expect 60-70% in fast mode, 80-85% in full mode
- **Plots**: 4 individual curves + 4 comparison plots

### Experiment 6.4 (Ablation Studies)
- **Learning rate ablation**: Tests 5 different learning rates for Adam
- **Beta parameter ablation**: Tests 4 different (β₁, β₂) combinations
- **Deep network**: Compares optimizers on 3-layer MLP
- **Plots**: 6 comparison plots showing sensitivity analysis

## Troubleshooting

### "MPS not available"
```bash
# Check if MPS is supported
python -c "import torch; print(torch.backends.mps.is_available())"

# If False, edit config.yaml:
device:
  use_mps: false  # Will use CPU instead
```

### "Out of memory"
Edit `config/config.yaml`:
```yaml
fast:
  batch_size: 64  # Reduce from 128
```

### "Training too slow"
- Ensure MPS is enabled (check config)
- Use fast mode instead of full mode
- Run experiments individually instead of all at once

### "Module not found"
```bash
# Reinstall requirements
pip install -r requirements.txt --upgrade
```

## Customization

### Add Your Own Optimizer

1. Create `optimizers/my_optimizer.py`
2. Implement following PyTorch Optimizer base class
3. Add to `optimizers/__init__.py`
4. Update config.yaml with hyperparameters
5. Add to experiment optimizer lists

### Modify Model Architecture

Edit files in `models/` directory:
- `logistic_regression.py`: Change layer sizes
- `mlp.py`: Add/remove hidden layers, change dropout
- `cnn.py`: Modify conv layers, add batch norm, etc.

### Change Datasets

Edit `data/data_loaders.py` to add new datasets:
- Follow similar pattern to MNIST/CIFAR-10 loaders
- Update config.yaml with dataset settings
- Adjust model input/output sizes accordingly

## Tips for Best Results

1. **Start with fast mode** to verify everything works
2. **Run one experiment** at a time to monitor progress
3. **Check plots** after each experiment before running next
4. **Switch to full mode** once satisfied with fast mode results
5. **Compare with paper figures** to validate implementation

## Next Steps

After running all experiments:

1. ✅ Compare your plots with Figure 1-4 from the Adam paper
2. ✅ Check if Adam consistently outperforms other optimizers
3. ✅ Verify learning rate and beta ablations show expected trends
4. ✅ Try full mode for paper-matching results
5. ✅ Experiment with different hyperparameters

## Support

For issues or questions:
- Check the main README.md for detailed documentation
- Review config/config.yaml for all available settings
- Inspect experiment files for implementation details

---

**Ready to start? Run:**
```bash
python run_all_experiments.py
```

**Happy experimenting! 🚀**

