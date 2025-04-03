import argparse
import os
import torch
from torch.utils.data import DataLoader
from transformers import (
    BertTokenizerFast,
    DataCollatorForLanguageModeling,
    get_scheduler,
    set_seed,
)
from datasets import load_from_disk
from tqdm import tqdm
import numpy as np
import time
import math
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

    def to_dict(self):
        """Convert configuration to a dictionary for saving"""
        return {k: v for k, v in self.__dict__.items()}

# The OntologicalTransformer model
class OntologicalTransformerModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Add layer normalization and dropout 
        self.layer_norm = torch.nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        
        self.encoder_layers = torch.nn.ModuleList(
            [self._create_layer() for _ in range(config.num_hidden_layers)]
        )
        
        self.pooler = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = torch.nn.Tanh()
        
        # Add a prediction head for masked language modeling
        self.cls = torch.nn.Linear(config.hidden_size, config.vocab_size)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize the weights similar to BERT initialization"""
        # Initialize embeddings
        self.embeddings.weight.data.normal_(mean=0.0, std=0.02)
        self.position_embeddings.weight.data.normal_(mean=0.0, std=0.02)
        
        # Initialize linear layers
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data.normal_(mean=0.0, std=0.02)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torch.nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        
    def _create_layer(self):
        config = self.config
        return torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.GELU(),
            torch.nn.Dropout(config.hidden_dropout_prob),
            torch.nn.Linear(config.hidden_size, config.hidden_size),
            torch.nn.Dropout(config.hidden_dropout_prob)
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
        
        # Apply layer normalization and dropout
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Extend attention mask for the ontological operations
            extended_attention_mask = attention_mask.unsqueeze(-1)
            extended_attention_mask = extended_attention_mask.to(dtype=hidden_states.dtype)
            hidden_states = hidden_states * extended_attention_mask
        
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
            # Only compute loss on masked tokens
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = prediction_scores.view(-1, self.config.vocab_size)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
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
        
    def save_pretrained(self, save_directory):
        """Save model and configuration to directory"""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save configuration
        config_dict = self.config.to_dict()
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            import json
            json.dump(config_dict, f, indent=2)
        
        # Save model weights
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        print(f"Model saved to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, pretrained_model_path):
        """Load model from pretrained weights"""
        import json
        
        # Load configuration
        with open(os.path.join(pretrained_model_path, "config.json"), "r") as f:
            config_dict = json.load(f)
        
        # Create configuration
        config = OntologicalTransformerConfig(**config_dict)
        
        # Create model
        model = cls(config)
        
        # Load weights
        state_dict = torch.load(os.path.join(pretrained_model_path, "pytorch_model.bin"), 
                               map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train OntologicalTransformer model on WikiText-2 dataset")
    parser.add_argument("--dataset_dir", type=str, required=True, 
                        help="Directory with preprocessed dataset")
    parser.add_argument("--output_dir", type=str, default="./output", 
                        help="Directory to save model")
    parser.add_argument("--pretrained_model_dir", type=str, default=None, 
                        help="Path to pretrained model to continue training")
    parser.add_argument("--tokenizer_name", type=str, default="bert-base-uncased", 
                        help="Tokenizer to use")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, 
                        help="Max steps to train for. Overrides num_train_epochs if positive.")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=500, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=1000, help="Saving checkpoints frequency")
    parser.add_argument("--mlm_probability", type=float, default=0.15, 
                        help="Probability for masked tokens")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run training on")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=6, help="Number of hidden layers")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set the seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = torch.device(args.device)
    
    # Load tokenizer
    tokenizer = BertTokenizerFast.from_pretrained(args.tokenizer_name)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading preprocessed dataset from {args.dataset_dir}...")
    dataset = load_from_disk(args.dataset_dir)
    
    # Load or create model
    if args.pretrained_model_dir:
        print(f"Loading pretrained model from {args.pretrained_model_dir}...")
        model = OntologicalTransformerModel.from_pretrained(args.pretrained_model_dir)
    else:
        print("Creating new OntologicalTransformer model...")
        config = OntologicalTransformerConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            max_position_embeddings=args.max_seq_length
        )
        model = OntologicalTransformerModel(config)
        print(f"Model configuration: {config}")
    
    model.to(device)
    
    # Get number of model parameters
    print(f"Model has {model.get_parameter_count():,} parameters")
    
    # Prepare data collator for masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability
    )
    
    # Create DataLoader
    train_dataloader = DataLoader(
        dataset["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=data_collator
    )
    
    # Prepare optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Determine number of training steps
    if args.max_steps > 0:
        num_training_steps = args.max_steps
        num_train_epochs = math.ceil(args.max_steps / len(train_dataloader))
    else:
        num_training_steps = len(train_dataloader) * args.num_train_epochs
        num_train_epochs = args.num_train_epochs
    
    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Setup training loop
    print(f"Starting training for {num_train_epochs} epochs ({num_training_steps} steps)...")
    progress_bar = tqdm(range(num_training_steps))
    
    # Training metrics
    global_step = 0
    total_loss = 0
    logging_loss = 0
    best_loss = float("inf")
    train_losses = []
    start_time = time.time()
    
    # Training loop
    model.train()
    for epoch in range(num_train_epochs):
        print(f"\nEpoch {epoch + 1}/{num_train_epochs}")
        
        for batch in train_dataloader:
            # Break if we reached max_steps
            if args.max_steps > 0 and global_step >= args.max_steps:
                break
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
                labels=batch["labels"]
            )
            
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Log metrics
            global_step += 1
            total_loss += loss.item()
            
            if global_step % args.logging_steps == 0:
                avg_loss = (total_loss - logging_loss) / args.logging_steps
                logging_loss = total_loss
                train_losses.append(avg_loss)
                
                print(f"Step {global_step}: loss = {avg_loss:.4f}")
                
                # Save best model
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            
            # Save checkpoint
            if args.save_steps > 0 and global_step % args.save_steps == 0:
                checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                model.save_pretrained(checkpoint_dir)
                print(f"Saved checkpoint to {checkpoint_dir}")
            
            progress_bar.update(1)
    
    # Save final model
    model.save_pretrained(args.output_dir)
    
    # Calculate and print training metrics
    train_time = time.time() - start_time
    avg_train_loss = total_loss / global_step
    
    print(f"\nTraining completed in {train_time:.2f} seconds")
    print(f"Average training loss: {avg_train_loss:.4f}")
    
    # Save training metrics
    with open(os.path.join(args.output_dir, "training_metrics.txt"), "w") as f:
        f.write(f"Training time: {train_time:.2f} seconds\n")
        f.write(f"Number of steps: {global_step}\n")
        f.write(f"Average training loss: {avg_train_loss:.4f}\n")
        f.write(f"Best training loss: {best_loss:.4f}\n")
    
    # Plot loss curve
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(args.logging_steps, global_step + 1, args.logging_steps), train_losses)
        plt.title("Training Loss")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.savefig(os.path.join(args.output_dir, "training_loss.png"))
        print(f"Training loss plot saved to {os.path.join(args.output_dir, 'training_loss.png')}")
    except Exception as e:
        print(f"Failed to generate training loss plot: {str(e)}")

if __name__ == "__main__":
    main() 