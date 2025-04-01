import argparse
import torch
from transformers import BertForMaskedLM, BertTokenizerFast
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with BERT model")
    parser.add_argument("--model_dir", type=str, required=True, help="Directory with the trained model")
    parser.add_argument("--text", type=str, required=True, help="Input text with [MASK] tokens")
    parser.add_argument("--top_k", type=int, default=5, help="Show top-k predictions for each mask")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to run inference on")
    return parser.parse_args()

def predict_masked_words(model, tokenizer, text, top_k=5, device="cpu"):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get the position of [MASK] token(s)
    mask_token_id = tokenizer.mask_token_id
    mask_positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)[1]
    
    if len(mask_positions) == 0:
        print("No [MASK] tokens found in the input text.")
        return []
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**inputs)
    
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
    
    # Load tokenizer and model
    print(f"Loading model from {args.model_dir}...")
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model = BertForMaskedLM.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()
    
    # Make sure the input has [MASK] token
    if tokenizer.mask_token not in args.text:
        print(f"Input must contain at least one {tokenizer.mask_token} token.")
        return
    
    # Highlight masks in the input text
    highlighted_text = highlight_masks(args.text, tokenizer)
    print(f"\nInput text: {highlighted_text}")
    
    # Run prediction
    results = predict_masked_words(model, tokenizer, args.text, args.top_k, args.device)
    
    # Print results
    print("\nPredictions:")
    for i, result in enumerate(results):
        print(f"\nMask {i+1}:")
        for j, pred in enumerate(result["predictions"]):
            print(f"  {j+1}. {pred['token']} (probability: {pred['probability']:.4f})")

if __name__ == "__main__":
    main() 