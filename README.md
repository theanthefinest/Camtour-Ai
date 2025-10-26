# Fine-Tuning with Mistral Model

A comprehensive guide for fine-tuning Mistral models with optimized training workflows.

## 📋 Prerequisites

> **⚠️ Important:** Local training is **not recommended**. For optimal performance and cost-effectiveness, use [Google Colab](https://colab.google/) with T4 GPU runtimes.

## 🚀 Setup Instructions

### 1. Environment Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
```

**Activate the environment:**
```bash
# For Linux/Mac users
source .venv/bin/activate

# For Windows users
source .venv/Scripts/activate
```

### 2. CUDA Installation (High-Performance Users)

For users with CUDA-compatible GPUs, install [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) first, then:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Install Bitsandbytes Wheel

> **💡 Tip:** Add the `--no-deps` flag if you want to avoid reinstalling existing dependencies.

```bash
pip install --force-reinstall 'https://github.com/bitsandbytes-foundation/bitsandbytes/releases/download/continuous-release_multi-backend-refactor/bitsandbytes-0.44.1.dev0-py3-none-win_amd64.whl'
```

### 5. Google Colab Setup

For Google Colab users, paste the following code at the top of your notebook and run:

```python
!pip install -U transformers
!pip install torch torchvision torchaudio
!pip install datasets==2.16.0
!pip install bitsandbytes
!pip install peft
!pip install trl
!pip install wandb
```

## 📂 Project Structure

- **`trainer.ipynb`** - Main fine-tuning workflow (start here)
- **`Trainer.py`** - Training augmentation examples with detailed comments
- **`requirements.txt`** - Python dependencies
- **`configure.ini`** - Configuration settings

## 🔧 Training Optimization

The training augmentation techniques and optimizations are documented with detailed comments in the notebooks. Refer to the notebooks for:

- Data preprocessing strategies
- Hyperparameter tuning
- Memory optimization techniques
- Training monitoring with Weights & Biases

## 🎯 Getting Started

1. **Start with `trainer.ipynb`** - This contains the final optimized fine-tuning process
2. Review the training augmentation examples in `camtour_fine_tuning.ipynb`
3. Adjust configuration parameters as needed

---

## 👨‍💻 Author

**LY Chhaythean** (Zhang Jiang)

*Happy fine-tuning! 🚀*
