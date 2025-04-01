import os
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import (
    BertForMaskedLM,
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer, 
    TrainingArguments,
    set_seed,
    BertConfig
)
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Train BERT model on WikiText-2 dataset")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save model")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model to fine-tune")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps to train for. Overrides num_train_epochs if positive.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="MLM probability")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--from_scratch", action="store_true", help="Train model from scratch instead of fine-tuning")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    # Initialize model - either from scratch or pre-trained
    if args.from_scratch:
        # Create a configuration for the model
        config = BertConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=768,
            num_hidden_layers=6,  # Reduced from 12 in BERT-base for faster training
            num_attention_heads=8,
            intermediate_size=2048,
            max_position_embeddings=args.max_seq_length,
            type_vocab_size=2,
        )
        
        # Initialize a model with random weights
        model = BertForMaskedLM(config)
        print("Initializing model from scratch with random weights")
    else:
        # Load pre-trained model
        model = BertForMaskedLM.from_pretrained(args.model_name)
        print(f"Loading pre-trained model: {args.model_name}")
    
    # Load dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
    
    # Tokenize dataset
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
    )
    
    # Set up data collator for masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f"{args.output_dir}/logs",
        logging_steps=100,
        save_steps=1000,
        save_total_limit=2,
        max_steps=args.max_steps,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
    )
    
    # Train the model
    trainer.train()
    
    # Save the model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main() 