import argparse
import torch
import numpy as np
from transformers import BertTokenizerFast
import matplotlib.pyplot as plt
import time
import os
from datetime import datetime

# Import our Ontological model classes from the evaluation script
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
    parser = argparse.ArgumentParser(description="Run inference with OntologicalTransformer model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with the model")
    parser.add_argument("--text", type=str, required=True, help="Input text with [MASK] tokens")
    parser.add_argument("--top_k", type=int, default=5, help="Show top-k predictions for each mask")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on")
    parser.add_argument("--vocab_size", type=int, default=30522, help="Vocabulary size")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size")
    parser.add_argument("--num_hidden_layers", type=int, default=6, help="Number of hidden layers")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    return parser.parse_args()

def predict_masked_words(model, tokenizer, text, top_k=5, device="cpu"):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    
    # Move to device and filter out keys not expected by the model
    # BERT tokenizer might add token_type_ids which our model doesn't accept
    filtered_inputs = {
        k: v.to(device) 
        for k, v in inputs.items() 
        if k in ['input_ids', 'attention_mask']  # Only keep keys that our model expects
    }
    
    # Get the position of [MASK] token(s)
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (filtered_inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1]
    
    if len(mask_positions) == 0:
        print("No [MASK] tokens found in the input text.")
        return []
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**filtered_inputs)
    
    results = []
    # Process each mask position
    for mask_pos in mask_positions:
        # Get the predictions for this mask position
        logits = outputs.logits[0, mask_pos, :]
        probs = torch.softmax(logits, dim=0)
        
        # Get top-k predictions
        top_k_probs, top_k_indices = torch.topk(probs, top_k)
        
        # Map indices to tokens
        top_tokens = [tokenizer.convert_ids_to_tokens(idx.item()) for idx in top_k_indices]
        top_probs = [prob.item() for prob in top_k_probs]
        
        # Add to results
        results.append({
            "position": mask_pos.item(),
            "predictions": [{"token": token, "probability": prob} for token, prob in zip(top_tokens, top_probs)]
        })
    
    return results

def highlight_masks(text, tokenizer):
    # Split text into words to identify mask positions
    words = text.split()
    mask_token = tokenizer.mask_token
    mask_indices = [i for i, word in enumerate(words) if mask_token in word]
    
    if not mask_indices:
        return text
    
    # Create highlighted text
    highlighted = text
    for i, idx in enumerate(mask_indices):
        # Replace mask token with numbered mask
        mask_in_text = words[idx]
        highlighted = highlighted.replace(mask_in_text, f"[MASK_{i+1}]", 1)
    
    return highlighted

def main():
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device)
    
    # Initialize tokenizer
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    # Create model configuration
    print("Initializing OntologicalTransformer model...")
    config = OntologicalTransformerConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        max_position_embeddings=args.max_seq_length
    )
    
    # Initialize model
    model = OntologicalTransformerModel(config)
    model.to(device)
    model.eval()
    
    # Make sure the input has [MASK] token
    if tokenizer.mask_token not in args.text:
        print(f"Input must contain at least one {tokenizer.mask_token} token.")
        return
    
    # Highlight masks in the input text
    highlighted_text = highlight_masks(args.text, tokenizer)
    print(f"\nInput text: {highlighted_text}")
    
    # Run prediction
    start_time = time.time()
    results = predict_masked_words(model, tokenizer, args.text, args.top_k, args.device)
    inference_time = time.time() - start_time
    
    # Print results
    print("\nPredictions from OntologicalTransformer:")
    for i, result in enumerate(results):
        print(f"\nMask {i+1}:")
        for j, pred in enumerate(result["predictions"]):
            print(f"  {j+1}. {pred['token']} (probability: {pred['probability']:.4f})")
    
    print(f"\nInference completed in {inference_time:.4f} seconds")
    print(f"Model has {model.get_parameter_count():,} parameters")

if __name__ == "__main__":
    main() 