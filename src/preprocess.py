import argparse
import os
from datasets import load_dataset
from transformers import BertTokenizerFast

def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess WikiText-2 dataset for BERT training")
    parser.add_argument("--output_dir", type=str, default="./data", help="Directory to save processed data")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Tokenizer to use")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--cache_dir", type=str, default="./data/cache", help="Cache directory for datasets")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)
    
    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    
    # Load WikiText-2 dataset
    print("Loading WikiText-2 dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", cache_dir=args.cache_dir)
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length,
            return_special_tokens_mask=True
        )
    
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )
    
    # Save processed dataset
    print("Saving preprocessed dataset...")
    tokenized_datasets.save_to_disk(os.path.join(args.output_dir, "processed_wikitext2"))
    
    # Print dataset stats
    print("\nDataset statistics:")
    print(f"Train set: {len(tokenized_datasets['train'])} examples")
    print(f"Validation set: {len(tokenized_datasets['validation'])} examples")
    print(f"Test set: {len(tokenized_datasets['test'])} examples")
    
    print(f"\nPreprocessed dataset saved to {os.path.join(args.output_dir, 'processed_wikitext2')}")

if __name__ == "__main__":
    main() 