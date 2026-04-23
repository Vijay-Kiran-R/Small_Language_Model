# SLM Quick Start Guide

Moving to a machine with a powerful GPU is the perfect next step! Follow these instructions to get your environment running smoothly.

## Step 1: Setup the Environment
Once you clone your project via Git onto the new machine, open a terminal in the project folder and install the dependencies.

```bash
# Optional but recommended: Create a virtual environment first
python -m venv slm

# Activate the virtual environment
# On Linux/macOS:
source slm/bin/activate  
# On Windows:
slm\Scripts\activate

# Install all required packages (PyTorch, transformers, datasets, pytest, etc.)
pip install -r requirements.txt
```

## Step 2: Test the Environment
Before you start a multi-hour training run, make sure the new machine is set up correctly by running the automated test suite.

```bash
pytest tests/
```
> [!NOTE]
> If all the tests pass, it means your GPU, PyTorch installation, and model architecture are working perfectly and are ready for heavy workloads.

## Step 3: Run the Training Pipeline
You can execute your project in chronological order just by running the Python scripts. All intermediate progress will be saved in the `trained_models/` directory.

### Pre-Training Phase
Start by training your base model on the raw data shards.
```bash
python pretrain_stage1.py
python pretrain_stage2.py
python pretrain_stage3.py
```

### Health Check
At any point during or after pretraining, verify the model is healthy and didn't collapse:
```bash
python health_check_pretrain.py
```

### Fine-Tuning & GRPO Phase
Once the pretraining is done, you can align your model to follow instructions and reason:
```bash
python finetune_4a_sft.py       # Supervised Fine-Tuning
python finetune_4b_cot.py       # Chain-of-Thought Fine-Tuning
python phase4b5_grpo.py         # Reinforcement Learning (GRPO)
python phase55_extend.py        # Long-Context Extension
```

> [!TIP]
> If you want to run a fast, fully automated test of the **entire** pipeline from pretraining to fine-tuning on a tiny dataset just to see it work end-to-end, you can run:
> ```bash
> python mini_e2e_pipeline.py
> ```


## Step 4: Adding Custom Datasets

It depends on whether you want to add data for **pre-training** (teaching the model general knowledge) or **fine-tuning** (teaching it to answer questions and follow instructions).

### Adding Pre-Training Data
If you want to add massive raw text datasets (like Wikipedia, books, or web crawls), you will edit this file: `slm_project/data/download_pretrain.py`.

Inside that file, you will see functions like `download_stage1()`. You can add any dataset from HuggingFace by simply adding a new `stream_tokenize_shard` block like this:

```python
stream_tokenize_shard(
    'your-huggingface-dataset-name', None,
    1_000_000_000, 'data/shards/stage1/your_dataset_name'  # Downloads 1 Billion tokens
)
```

### Adding Fine-Tuning Data
If you want to add conversational data (Q&A, chat logs, reasoning examples), you will edit the specific fine-tuning scripts in your root directory:

- `finetune_4a_sft.py` (For general conversation/chat datasets)
- `finetune_4b_cot.py` (For Chain-of-Thought math/reasoning datasets)
- `finetune_4c_domain.py` (For specific domains like medicine or coding)

If you open `finetune_4a_sft.py`, scroll to the very bottom and you will see a `# TODO: load SFT dataset` marker. That is exactly where you will wire up your custom datasets using the standard HuggingFace `load_dataset` function before running the `trainer.train_step()` loop!

## Step 5: How to Monitor Training Health

While your model is training for hours or days, you need to ensure it hasn't crashed or collapsed. You can monitor it in three ways:

### 1. Live Terminal Output
As the script runs, it will print out the current `Step`, `Loss`, and `Learning Rate`. You want to see the **Loss** steadily decreasing over time.

### 2. The Log File
All training metrics are permanently saved to a file called `trained_models/training_logs.jsonl`. 
Even if you close your terminal, you can open this file (or write a quick Python script to graph it) to view historical data for:
- Loss
- Gradient Norms (if this spikes to infinity, the model has collapsed)
- AttnRes Pseudo-Query Norms

### 3. The Health Check Script
At any point during the pre-training process (even while it's currently running!), you can open a second terminal and run:
```bash
python health_check_pretrain.py
```
This script automatically grabs the latest saved checkpoint from `trained_models/` and deeply analyzes it. It will check for NaN values (math errors), verify layer outputs, and even try to generate a token to ensure the model is actually functioning!