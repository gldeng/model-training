import argparse
import os
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Run the full BERT training pipeline on WikiText-2")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save model and outputs")
    parser.add_argument("--data_dir", type=str, default="./data", help="Directory to save processed data")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased", help="Model to fine-tune")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Max steps to train for. Overrides num_train_epochs if positive.")
    parser.add_argument("--max_seq_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--skip_preprocessing", action="store_true", help="Skip preprocessing step")
    parser.add_argument("--skip_training", action="store_true", help="Skip training step")
    parser.add_argument("--skip_evaluation", action="store_true", help="Skip evaluation step")
    parser.add_argument("--test_inference", action="store_true", help="Run inference test after training")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (cuda or cpu)")
    parser.add_argument("--top_k", type=int, default=5, help="Top k predictions to show in inference")
    parser.add_argument("--from_scratch", action="store_true", help="Train model from scratch instead of fine-tuning")
    parser.add_argument("--use_ontological", action="store_true", help="Use OntologicalTransformer model instead of BERT")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size for ontological model")
    parser.add_argument("--num_hidden_layers", type=int, default=6, help="Number of hidden layers for ontological model")
    parser.add_argument("--plot", action="store_true", help="Generate performance plots for evaluation")
    parser.add_argument("--mlm_probability", type=float, default=0.15, help="Probability for masked tokens")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--logging_steps", type=int, default=500, help="Logging frequency")
    parser.add_argument("--save_steps", type=int, default=1000, help="Saving checkpoints frequency")
    return parser.parse_args()

def run_command(cmd, description):
    print(f"\n{'='*80}")
    print(f"Running {description}...")
    print(f"{'='*80}\n")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            print(line, end='')
        process.wait()
        
        if process.returncode != 0:
            print(f"\nERROR: {description} failed with exit code {process.returncode}")
            return False
        
        print(f"\n{description} completed successfully!")
        return True
    except Exception as e:
        print(f"\nERROR: {description} failed with exception: {str(e)}")
        return False

def main():
    args = parse_args()
    
    # Create directories if they don't exist
    for directory in [args.output_dir, args.data_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    # Build common command arguments for preprocessing and training
    preprocess_train_args = [
        f"--model_name={args.model_name}",
        f"--max_seq_length={args.max_seq_length}",
    ]
    
    # Build common command arguments for evaluation and inference (these don't use model_name)
    eval_inference_args = []
    
    if args.device:
        preprocess_train_args.append(f"--device={args.device}")
        eval_inference_args.append(f"--device={args.device}")
    
    # Step 1: Preprocess data
    if not args.skip_preprocessing:
        preprocess_cmd = [
            sys.executable, "src/preprocess.py",
            f"--output_dir={args.data_dir}",
        ] + preprocess_train_args
        
        if not run_command(preprocess_cmd, "Data Preprocessing"):
            print("Exiting due to preprocessing failure.")
            return 1
    
    # Step 2: Train model
    if not args.skip_training:
        if args.use_ontological:
            # Use the ontological training script
            train_cmd = [
                sys.executable, "src/train_ontological.py",
                f"--dataset_dir={os.path.join(args.data_dir, 'processed_wikitext2')}",
                f"--output_dir={args.output_dir}",
                f"--tokenizer_name={args.model_name}",
                f"--batch_size={args.batch_size}",
                f"--learning_rate={args.learning_rate}",
                f"--num_train_epochs={args.num_train_epochs}",
                f"--max_steps={args.max_steps}",
                f"--hidden_size={args.hidden_size}",
                f"--num_hidden_layers={args.num_hidden_layers}",
                f"--max_seq_length={args.max_seq_length}",
                f"--mlm_probability={args.mlm_probability}",
                f"--warmup_steps={args.warmup_steps}",
                f"--weight_decay={args.weight_decay}",
                f"--logging_steps={args.logging_steps}",
                f"--save_steps={args.save_steps}",
            ]
            
            if args.device:
                train_cmd.append(f"--device={args.device}")
                
            if not run_command(train_cmd, "OntologicalTransformer Training"):
                print("Exiting due to training failure.")
                return 1
        else:
            train_cmd = [
                sys.executable, "src/train.py",
                f"--output_dir={args.output_dir}",
                f"--batch_size={args.batch_size}",
                f"--learning_rate={args.learning_rate}",
                f"--num_train_epochs={args.num_train_epochs}",
                f"--max_steps={args.max_steps}",
            ] + preprocess_train_args
            
            # Add from_scratch flag if specified
            if args.from_scratch:
                train_cmd.append("--from_scratch")
            
            if not run_command(train_cmd, "Model Training"):
                print("Exiting due to training failure.")
                return 1
    
    # Step 3: Evaluate model
    if not args.skip_evaluation:
        if args.use_ontological:
            # Use the ontological evaluation script
            eval_cmd = [
                sys.executable, "src/evaluate_ontological.py",
                f"--model_dir={args.output_dir}",
                f"--dataset_dir={os.path.join(args.data_dir, 'processed_wikitext2')}",
                f"--batch_size={args.batch_size}",
                f"--hidden_size={args.hidden_size}",
                f"--num_hidden_layers={args.num_hidden_layers}",
            ]
            
            if args.plot:
                eval_cmd.append("--plot")
                
            eval_cmd += eval_inference_args
        else:
            # Use the standard BERT evaluation script
            eval_cmd = [
                sys.executable, "src/evaluate.py",
                f"--model_dir={args.output_dir}",
                f"--dataset_dir={os.path.join(args.data_dir, 'processed_wikitext2')}",
                f"--batch_size={args.batch_size}",
            ] + eval_inference_args
        
        if not run_command(eval_cmd, "Model Evaluation"):
            print("WARNING: Evaluation failed, but continuing with pipeline.")
    
    # Step 4: Run inference test
    if args.test_inference:
        test_text = "The capital of France is [MASK]."
        
        if args.use_ontological:
            # Use the ontological inference script
            inference_cmd = [
                sys.executable, "src/inference_ontological.py",
                f"--model_dir={args.output_dir}",
                f"--text={test_text}",
                f"--top_k={args.top_k}",
                f"--hidden_size={args.hidden_size}",
                f"--num_hidden_layers={args.num_hidden_layers}",
            ] + eval_inference_args
        else:
            # Use the standard BERT inference script
            inference_cmd = [
                sys.executable, "src/inference.py",
                f"--model_dir={args.output_dir}",
                f"--text={test_text}",
                f"--top_k={args.top_k}",
            ] + eval_inference_args
        
        if not run_command(inference_cmd, "Inference Test"):
            print("WARNING: Inference test failed.")
    
    print("\nPipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 