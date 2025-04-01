# BERT Training on WikiText-2 Dataset

This project provides scripts to train a BERT-base model on the WikiText-2 dataset for masked language modeling.

## Project Structure

```
.
├── src/
│   ├── main.py         # Main script to run the full pipeline
│   ├── preprocess.py   # Preprocess WikiText-2 dataset
│   ├── train.py        # Train BERT model
│   ├── evaluate.py     # Evaluate trained model
│   └── inference.py    # Run inference with trained model
├── data/               # Directory for processed data
├── output/             # Directory for model outputs
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Requirements

- Python 3.7+
- PyTorch 1.10+
- Transformers 4.18+
- Datasets 2.0+
- Accelerate 0.26+
- CUDA-compatible GPU (recommended but optional)

## Installation

```bash
# Clone this repository
git clone <repository-url>
cd <repository-directory>

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Full Pipeline

To run the entire pipeline (preprocessing, training, evaluation, and inference test):

```bash
python src/main.py --output_dir ./output --data_dir ./data --num_train_epochs 3
```

For quicker testing, you can limit the number of training steps:

```bash
python src/main.py --output_dir ./output --data_dir ./data --max_steps 100 --test_inference
```

### Step-by-Step Execution

#### 1. Preprocess the WikiText-2 Dataset

```bash
python src/preprocess.py --output_dir ./data
```

This will download the WikiText-2 dataset, tokenize it, and save the processed data to the specified directory.

#### 2. Train the BERT Model

```bash
python src/train.py --output_dir ./output --model_name bert-base-uncased --batch_size 8 --num_train_epochs 3
```

For quicker training, you can use the `--max_steps` parameter:

```bash
python src/train.py --output_dir ./output --max_steps 100
```

#### 3. Evaluate the Model

```bash
python src/evaluate.py --model_dir ./output --dataset_dir ./data/processed_wikitext2
```

#### 4. Run Inference

```bash
python src/inference.py --model_dir ./output --text "The capital of France is [MASK]."
```

## Command-Line Arguments

Common arguments available in all scripts:

- `--model_name`: Pre-trained model to use (default: "bert-base-uncased")
- `--max_seq_length`: Maximum sequence length (default: 128)
- `--batch_size`: Batch size for training/evaluation (default: 8)
- `--device`: Device to run on ("cuda" or "cpu"), defaults to CUDA if available

See each script's help for more specific arguments:

```bash
python src/<script>.py --help
```

## Custom Parameters

To customize the training process, modify these parameters:

- `--learning_rate`: Learning rate (default: 5e-5)
- `--num_train_epochs`: Number of training epochs (default: 3)
- `--max_steps`: Maximum number of training steps, overrides epochs if positive (default: -1)
- `--mlm_probability`: Probability for masked tokens (default: 0.15)

## Examples

### Training with Custom Parameters

```bash
python src/main.py --batch_size 16 --learning_rate 3e-5 --num_train_epochs 5 --max_seq_length 256
```

### Skipping Steps in the Pipeline

```bash
# Skip preprocessing if you've already processed the data
python src/main.py --skip_preprocessing

# Evaluate only (skip preprocessing and training)
python src/main.py --skip_preprocessing --skip_training
```

### Running Inference with Custom Text

```bash
python src/inference.py --model_dir ./output --text "The [MASK] runs quickly across the field." --top_k 10
```

## License

[Insert Your License Information Here] 