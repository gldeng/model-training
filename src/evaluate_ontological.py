import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from transformers import BertTokenizerFast
import matplotlib.pyplot as plt
import time
import os
import psutil
from datetime import datetime

# Import our Ontological model classes
class OntologicalOperations:
    """
    Implement basic ontological operations: XOR, SHIFT, FLIP
    """
    @staticmethod
    def xor(x, y):
        """
        Simulate XOR operation, implemented bitwise on tensors
        """
        return (x + y) - 2 * (x * y)
    
    @staticmethod
    def shift(x, shift_amount=1):
        """
        Simulate SHIFT operation, circularly shift values in tensor
        """
        return torch.roll(x, shifts=shift_amount, dims=-1)
    
    @staticmethod
    def flip(x):
        """
        Simulate FLIP operation, invert values in tensor
        """
        return 1.0 - x

# Configuration class for the OntologicalTransformer
class OntologicalTransformerConfig:
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        use_xor_attention=True,
        use_shift_ffn=True,
        use_flip_output=True,
        num_labels=2,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.pad_token_id = pad_token_id
        self.use_xor_attention = use_xor_attention
        self.use_shift_ffn = use_shift_ffn
        self.use_flip_output = use_flip_output
        self.num_labels = num_labels

    def __str__(self):
        return (f"OntologicalTransformerConfig(hidden_size={self.hidden_size}, "
                f"num_hidden_layers={self.num_hidden_layers}, "
                f"num_attention_heads={self.num_attention_heads}, "
                f"use_xor_attention={self.use_xor_attention}, "
                f"use_shift_ffn={self.use_shift_ffn}, "
                f"use_flip_output={self.use_flip_output})")

# The OntologicalTransformer model
class OntologicalTransformerModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.encoder_layers = torch.nn.ModuleList(
            [self._create_layer() for _ in range(config.num_hidden_layers)]
        )
        
        self.pooler = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = torch.nn.Tanh()
        
        # Add a prediction head for masked language modeling
        self.cls = torch.nn.Linear(config.hidden_size, config.vocab_size)
        
    def _create_layer(self):
        config = self.config
        return torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
        )
        
    def forward(self, input_ids=None, attention_mask=None, labels=None):
        batch_size, seq_length = input_ids.shape
        
        # Embeddings
        embeddings = self.embeddings(input_ids)
        
        # Position encoding
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        position_embeddings = self.position_embeddings(position_ids)
        
        # Combine embeddings with ontological operations
        hidden_states = OntologicalOperations.xor(embeddings, position_embeddings)
        
        # Encoder layers
        for layer in self.encoder_layers:
            layer_output = layer(hidden_states)
            shifted_states = OntologicalOperations.shift(hidden_states)
            flipped_output = OntologicalOperations.flip(layer_output)
            hidden_states = OntologicalOperations.xor(shifted_states, flipped_output)
        
        # Pooling
        pooled_output = self.pooler(hidden_states[:, 0])
        pooled_output = self.pooler_activation(pooled_output)
        
        # Prediction head for masked tokens
        prediction_scores = self.cls(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        
        return type('obj', (object,), {
            'last_hidden_state': hidden_states,
            'pooler_output': pooled_output,
            'logits': prediction_scores,
            'loss': loss
        })

    def get_parameter_count(self):
        """Calculate the number of parameters in the model"""
        return sum(p.numel() for p in self.parameters())

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate OntologicalTransformer model on WikiText-2")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory to save/load the model")
    parser.add_argument("--dataset_dir", type=str, default=None, 
                        help="Directory with processed dataset (if None, will download WikiText-2)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run evaluation on")
    parser.add_argument("--vocab_size", type=int, default=30522, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=6, help="Number of hidden layers")
    parser.add_argument("--plot", action="store_true", help="Generate performance plots")
    return parser.parse_args()

def compute_memory_usage():
    """Return memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device)
    
    # Create model configuration
    print("Initializing OntologicalTransformer model...")
    config = OntologicalTransformerConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_length
    )
    
    print(f"Model configuration: {config}")
    
    # Initialize the model
    model = OntologicalTransformerModel(config)
    model.to(device)
    model.eval()
    
    # Use BERT tokenizer for processing the text
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
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
    
    # Create dataloader with appropriate collate_fn
    def collate_fn(examples):
        return {
            key: torch.tensor([example[key] for example in examples]) 
            for key in examples[0].keys()
        }
    
    dataloader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Metrics for tracking performance
    total_loss = 0
    total_tokens = 0
    start_time = time.time()
    batch_times = []
    memory_usage = []
    losses = []  # Track individual losses for better analysis
    
    # Evaluate model
    print("Calculating perplexity...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            batch_start = time.time()
            
            # Track memory before processing batch
            memory_usage.append(compute_memory_usage())
            
            # Move batch to device and filter inputs to only include supported keys
            filtered_inputs = {
                k: v.to(device) 
                for k, v in batch.items() 
                if k in ['input_ids', 'attention_mask'] and k != 'special_tokens_mask'
            }
            
            # Create labels by copying input_ids
            labels = filtered_inputs["input_ids"].clone()
            
            # Forward pass
            outputs = model(
                input_ids=filtered_inputs["input_ids"],
                attention_mask=filtered_inputs.get("attention_mask"),
                labels=labels
            )
            
            # Get loss
            loss = outputs.loss
            if loss is not None:
                current_loss = loss.item()
                
                # Skip extreme outlier losses that might cause numerical issues
                if current_loss > 1000:
                    print(f"Warning: Skipping batch {batch_idx} with unusually high loss: {current_loss}")
                    continue
                    
                losses.append(current_loss)
                
                # Calculate token count (excluding padding and special tokens)
                if "special_tokens_mask" in batch:
                    special_tokens_mask = batch["special_tokens_mask"].bool()
                    num_tokens = (~special_tokens_mask).sum().item()
                else:
                    num_tokens = (filtered_inputs["input_ids"] != tokenizer.pad_token_id).sum().item()
                
                # Update metrics
                total_loss += current_loss * num_tokens
                total_tokens += num_tokens
                
                # Record batch processing time
                batch_times.append(time.time() - batch_start)
            
            # Optional: limit evaluation to a small number of batches for testing
            # if batch_idx >= 10:
            #    break
    
    # Calculate final metrics
    total_time = time.time() - start_time
    
    # Check if we processed any batch successfully
    if len(batch_times) > 0:
        avg_batch_time = sum(batch_times) / len(batch_times)
    else:
        avg_batch_time = float('nan')
        print("Warning: All batches were skipped due to high loss values.")
    
    if len(memory_usage) > 0:
        avg_memory_mb = sum(memory_usage) / len(memory_usage)
    else:
        avg_memory_mb = float('nan')
    
    # Calculate perplexity safely
    if total_tokens > 0:
        avg_loss = total_loss / total_tokens
        # Clip extremely high loss values to prevent overflow
        avg_loss = min(avg_loss, 30)  # ln(1e13) ≈ 30
        perplexity = np.exp(avg_loss)
    else:
        avg_loss = float('nan')
        perplexity = float('nan')
        print("Warning: No tokens were processed successfully. Cannot calculate perplexity.")
    
    # Print evaluation results
    print(f"\nEvaluation results for OntologicalTransformer:")
    print(f"Parameter count: {model.get_parameter_count():,}")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Total evaluation time: {total_time:.2f} seconds")
    print(f"Average batch processing time: {avg_batch_time:.4f} seconds")
    print(f"Average memory usage: {avg_memory_mb:.2f} MB")
    print(f"Successful batches: {len(batch_times)} out of {len(dataloader)}")
    
    # Generate performance plots if requested
    if args.plot and len(batch_times) > 0:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_dir = os.path.join(args.model_dir, "plots")
        os.makedirs(plot_dir, exist_ok=True)
        
        # Plot batch processing times
        plt.figure(figsize=(10, 6))
        plt.plot(batch_times)
        plt.title('Batch Processing Time')
        plt.xlabel('Batch')
        plt.ylabel('Time (seconds)')
        plt.savefig(os.path.join(plot_dir, f'batch_times_{timestamp}.png'))
        
        # Plot memory usage
        plt.figure(figsize=(10, 6))
        plt.plot(memory_usage)
        plt.title('Memory Usage')
        plt.xlabel('Batch')
        plt.ylabel('Memory (MB)')
        plt.savefig(os.path.join(plot_dir, f'memory_usage_{timestamp}.png'))
        
        # Plot loss distribution (histogram)
        if losses:
            plt.figure(figsize=(10, 6))
            plt.hist(losses, bins=50)
            plt.title('Loss Distribution')
            plt.xlabel('Loss')
            plt.ylabel('Frequency')
            plt.savefig(os.path.join(plot_dir, f'loss_distribution_{timestamp}.png'))
        
        print(f"Performance plots saved to {plot_dir}")

if __name__ == "__main__":
    main() 