# Adam Paper Experiments Replication

This repository contains a complete, modular implementation to replicate all experiments from **Section 6** of the paper:

> **"Adam: A Method for Stochastic Optimization"**  
> by Diederik P. Kingma and Jimmy Lei Ba (2014)  
> [https://arxiv.org/pdf/1412.6980](https://arxiv.org/pdf/1412.6980)

## 📋 Overview

This project replicates the following experiments:
- **Experiment 6.1**: Logistic Regression on MNIST
- **Experiment 6.2**: Multi-Layer Perceptron on MNIST
- **Experiment 6.3**: Convolutional Neural Network on CIFAR-10
- **Experiment 6.4**: Advanced Experiments and Ablation Studies

## 🏗️ Project Structure

```
AdamSGD/
├── config/
│   └── config.yaml                 # Hyperparameters and settings
├── optimizers/
│   ├── adam.py                     # Adam optimizer (Algorithm 1 from paper)
│   ├── sgd_momentum.py             # SGD with momentum
│   ├── adagrad.py                  # AdaGrad optimizer
│   └── rmsprop.py                  # RMSProp optimizer
├── models/
│   ├── logistic_regression.py     # Logistic regression model
│   ├── mlp.py                      # Multi-layer perceptron
│   └── cnn.py                      # Convolutional neural network
├── data/
│   └── data_loaders.py            # MNIST and CIFAR-10 loaders (auto-download)
├── utils/
│   ├── training.py                # Training utilities
│   ├── evaluation.py              # Evaluation metrics
│   └── plotting.py                # Visualization utilities
├── experiments/
│   ├── exp_6_1_logistic_regression.py
│   ├── exp_6_2_mlp.py
│   ├── exp_6_3_cnn.py
│   └── exp_6_4_advanced.py
├── results/                        # Generated plots and results
├── requirements.txt
└── run_all_experiments.py         # Master script
```

## 🚀 Quick Start

### 1. Setup Environment

First, activate your virtual environment and install dependencies:

```bash
cd AdamSGD
source adamVenv/bin/activate  # Already created
pip install -r requirements.txt
```

### 2. Run All Experiments

To run all experiments at once:

```bash
python run_all_experiments.py
```

### 3. Run Individual Experiments

You can also run experiments individually:

```bash
# Experiment 6.1: Logistic Regression on MNIST
python experiments/exp_6_1_logistic_regression.py

# Experiment 6.2: MLP on MNIST
python experiments/exp_6_2_mlp.py

# Experiment 6.3: CNN on CIFAR-10
python experiments/exp_6_3_cnn.py

# Experiment 6.4: Advanced experiments and ablations
python experiments/exp_6_4_advanced.py
```

### 4. Run Specific Experiments

```bash
# Run only experiments 6.1 and 6.2
python run_all_experiments.py --experiments 6.1 6.2

# Run with custom config
python run_all_experiments.py --config path/to/config.yaml
```

## ⚙️ Configuration

Edit `config/config.yaml` to customize experiments:

### Key Settings

**Mode**: Switch between `fast` (quick iteration) and `full` (paper-matching) mode
```yaml
mode: 'fast'  # or 'full'
```

**Device**: Configure for Apple Silicon (MPS), CUDA, or CPU
```yaml
device:
  use_mps: true   # Apple Silicon GPU
  use_cuda: false
```

**Optimizer Hyperparameters**: 
- Adam: α=0.001, β₁=0.9, β₂=0.999, ε=1e-8 (paper defaults)
- Can customize learning rates for each optimizer

**Training Settings**:
- Epochs (fast mode): LogReg=10, MLP=20, CNN=30
- Epochs (full mode): LogReg=100, MLP=100, CNN=200
- Batch size: 128 (configurable)

## 📊 Results and Plots

After running experiments, results are saved in:

```
results/
├── exp_6_1_logistic_regression/
│   ├── comparison_train_loss.png
│   ├── comparison_val_loss.png
│   ├── comparison_train_acc.png
│   ├── comparison_val_acc.png
│   └── [optimizer]_curves.png (for each optimizer)
├── exp_6_2_mlp/
│   └── [similar structure]
├── exp_6_3_cnn/
│   └── [similar structure]
└── exp_6_4_advanced/
    ├── lr_ablation_val_loss.png
    ├── lr_ablation_val_acc.png
    ├── beta_ablation_val_loss.png
    ├── deep_network_train_loss.png
    └── deep_network_val_acc.png
```

## 🧪 Experiments Details

### Experiment 6.1: Logistic Regression on MNIST
- **Model**: Single linear layer (784 → 10)
- **Optimizers**: Adam, SGD, SGD+Momentum, AdaGrad, RMSProp
- **Dataset**: MNIST (60k train, 10k test)
- **Objective**: Compare optimizer convergence on simple convex problem

### Experiment 6.2: Multi-Layer Perceptron on MNIST
- **Model**: 2 hidden layers (784 → 1000 → 1000 → 10) with ReLU and dropout
- **Optimizers**: Adam, SGD+Momentum, AdaGrad, RMSProp
- **Dataset**: MNIST
- **Objective**: Compare on deeper non-convex problem

### Experiment 6.3: CNN on CIFAR-10
- **Model**: 3 conv blocks + 3 FC layers with dropout
- **Optimizers**: Adam, SGD+Momentum, AdaGrad, RMSProp
- **Dataset**: CIFAR-10 (50k train, 10k test)
- **Objective**: Compare on vision task with spatial structure

### Experiment 6.4: Advanced Experiments
- **Learning Rate Ablation**: Test Adam with different learning rates
- **Beta Parameter Ablation**: Test different β₁ and β₂ values
- **Deep Network**: 3 hidden layers comparison
- **Objective**: Sensitivity analysis and robustness testing

## 🔬 Implementation Details

### Adam Optimizer
Implements Algorithm 1 from the paper:

```python
m_t = β₁ · m_{t-1} + (1 - β₁) · g_t
v_t = β₂ · v_{t-1} + (1 - β₂) · g_t²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ_t = θ_{t-1} - α · m̂_t / (√v̂_t + ε)
```

### Key Features
- ✅ Modular, clean code structure
- ✅ MPS backend support for Apple Silicon
- ✅ Automatic dataset downloading
- ✅ Configurable hyperparameters
- ✅ Comprehensive plotting matching paper figures
- ✅ Progress bars and logging
- ✅ Fast and full experiment modes

## 📈 Expected Results

Based on the paper, you should observe:
- **Adam** converges faster and achieves better final performance
- **SGD** requires careful learning rate tuning
- **AdaGrad** works well initially but learning rate decays too aggressively
- **RMSProp** performs well but Adam's bias correction helps
- **Learning rate ablation** shows Adam is robust to different learning rates

## 🖥️ System Requirements

- Python 3.9+
- PyTorch 2.0+
- 8GB+ RAM
- GPU recommended but not required
  - Apple Silicon: Uses MPS backend
  - NVIDIA: Set `use_cuda: true` in config
  - CPU: Falls back automatically

## 📝 Notes

- **Fast mode** is recommended for initial testing (10-30 minutes total)
- **Full mode** matches paper settings but takes longer (2-6 hours depending on hardware)
- Datasets are automatically downloaded to `data/` directory
- All random seeds are fixed for reproducibility (seed=42)

## 🔍 Troubleshooting

**Issue**: MPS not available
```bash
# Check PyTorch MPS support
python -c "import torch; print(torch.backends.mps.is_available())"
```

**Issue**: Out of memory
- Reduce batch size in config.yaml
- Use CPU instead of MPS/CUDA
- Run experiments individually instead of all at once

**Issue**: Slow training
- Ensure MPS backend is enabled in config
- Use fast mode for quicker iteration
- Reduce number of epochs

## 📚 References

1. Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.
2. Duchi, J., Hazan, E., & Singer, Y. (2011). Adaptive subgradient methods for online learning and stochastic optimization. JMLR.
3. Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient by a running average of its recent magnitude. COURSERA: Neural networks for machine learning.

## 📄 License

This implementation is for educational and research purposes, replicating the experiments from the Adam paper.

## 🙏 Acknowledgments

This implementation replicates the experiments from:
- Paper: "Adam: A Method for Stochastic Optimization" by Kingma & Ba
- Paper URL: https://arxiv.org/pdf/1412.6980

---

**Happy Experimenting! 🚀**
