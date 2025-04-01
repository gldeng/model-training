import argparse
import torch
import numpy as np
from transformers import BertForMaskedLM, BertTokenizerFast
from datasets import load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BERT model on WikiText-2 dataset")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with the trained model")
    parser.add_argument("--dataset_dir", type=str, default=None, 
                        help="Directory with processed dataset (if None, will download WikiText-2)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run evaluation on")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load tokenizer and model
    print(f"Loading model from {args.model_dir}...")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model = BertForMaskedLM.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()
    
    # Load dataset
    if args.dataset_dir:
        print(f"Loading preprocessed dataset from {args.dataset_dir}...")
        dataset = load_from_disk(args.dataset_dir)
        test_dataset = dataset["test"]
    else:
        print("Loading and processing WikiText-2 dataset...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
        
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
            remove_columns=["text"],
        )
        
        test_dataset = tokenized_datasets["test"]
    
    # Create dataloader
    dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate perplexity
    total_loss = 0
    total_tokens = 0
    
    print("Calculating perplexity...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move batch to device
            inputs = {k: v.to(args.device) for k, v in batch.items() if k != "special_tokens_mask"}
            
            # Forward pass
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss
            
            # Calculate token count (excluding padding and special tokens)
            if "special_tokens_mask" in batch:
                special_tokens_mask = batch["special_tokens_mask"].bool()
                num_tokens = (~special_tokens_mask).sum().item()
            else:
                num_tokens = (inputs["input_ids"] != tokenizer.pad_token_id).sum().item()
            
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
    
    # Calculate perplexity
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    
    print(f"\nEvaluation results:")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

if __name__ == "__main__":
    main() 