import numpy as np
import evaluate
import os
import torch
import traceback  # Add missing import for traceback
from transformers import TrainerCallback  # Add for custom callback
from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    PeftConfig,
    PeftModel
)
from sklearn.preprocessing import LabelEncoder
from utils.load_datasets import load_MR, load_Semeval2017A
import shutil
import sys

# Disable PyTorch 2.0 compiler to avoid Windows-specific CUDA issues
import torch._dynamo
torch._dynamo.config.suppress_errors = True
torch._dynamo.disable()  # Explicitly disable dynamo
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Prevent memory fragmentation

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_metrics(eval_pred):
    # Load metrics from evaluate library
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    recall_metric = evaluate.load("recall")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Calculate metrics
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    recall = recall_metric.compute(
        predictions=predictions, references=labels, average="macro"
    )

    return {
        "accuracy": accuracy["accuracy"],
        "f1": f1["f1"],
        "recall": recall["recall"],
    }


def prepare_dataset(X, y):
    texts, labels = [], []
    for text, label in zip(X, y):
        texts.append(text)
        labels.append(label)
    return Dataset.from_dict({"text": texts, "label": labels})


def get_model_specific_modules(model_name, model):
    """Get the correct target modules for each model architecture"""
    
    # Special cases for known problematic models
    if "siebert/sentiment-roberta-large-english" in model_name:
        return ["dense"]  # Use only dense layers for this model
        
    # Always go by model_name first (most reliable)
    if "distilbert" in model_name.lower():
        return ["q_lin", "k_lin", "v_lin", "out_lin", "lin1", "lin2"]
    elif "roberta" in model_name.lower():
        return ["query", "key", "value", "dense", "output.dense"]
    elif "bert" in model_name.lower() and "deberta" not in model_name.lower():
        return ["query", "key", "value", "dense"]
    elif "deberta" in model_name.lower():
        return ["query_proj", "key_proj", "value_proj", "dense"]
    elif "t5" in model_name.lower():
        return ["q", "k", "v", "o", "wi", "wo"]
    elif "gpt" in model_name.lower():
        return ["c_attn", "c_proj", "c_fc"]
        
    # If we couldn't identify by name, try to inspect the model directly
    print("Could not identify model type by name. Attempting to find valid modules...")
    
    # Get all named modules
    named_modules = dict(model.named_modules())
    
    # Try to find attention modules
    attention_candidates = []
    
    # Check for different common naming patterns
    for name in named_modules.keys():
        # Find attention query/key/value names
        if "query" in name or "q_proj" in name or "q_lin" in name:
            attention_candidates.append(name.split(".")[-1])
        elif "key" in name or "k_proj" in name or "k_lin" in name:
            attention_candidates.append(name.split(".")[-1])
        elif "value" in name or "v_proj" in name or "v_lin" in name:
            attention_candidates.append(name.split(".")[-1])
        elif "attn" in name and ("out" in name or "o_proj" in name or "output" in name):
            attention_candidates.append(name.split(".")[-1])
    
    # If we found attention modules, use them
    if attention_candidates:
        return list(set(attention_candidates))
    
    # Last resort: check for classifier/output layers
    for name in named_modules.keys():
        if "classifier" in name:
            return ["classifier"]
        elif "out_proj" in name:
            return ["out_proj"]
    
    # Absolute fallback - try these common names
    return ["dense", "attention", "classifier"]


class SkipBestModelLoadingCallback(TrainerCallback):
    """Callback to skip best model loading at the end of training, which often fails with LoRA"""
    def on_train_end(self, args, state, control, **kwargs):
        # Save a checkpoint of the final model before best model loading
        if "model" in kwargs:
            try:
                # Get model and save it with a special name
                model = kwargs["model"]
                final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
                os.makedirs(final_checkpoint_dir, exist_ok=True)
                model.save_pretrained(final_checkpoint_dir)
                print(f"Saved final model checkpoint to {final_checkpoint_dir}")
            except Exception as e:
                print(f"Failed to save final checkpoint: {e}")
        
        # Don't prevent the trainer's default behaviors
        return control


def safe_save_model(model, tokenizer, save_path, use_lora=True):
    """
    Safely save the model and tokenizer, with additional error handling
    
    Args:
        model: The trained model
        tokenizer: The tokenizer used
        save_path: Path to save the model
        use_lora: Whether LoRA was used
    """
    try:
        # Ensure directory exists
        os.makedirs(save_path, exist_ok=True)
        
        if use_lora:
            try:
                # Try to merge and save the base model with LoRA weights
                print("Attempting to merge and save model with LoRA weights...")
                merged_model = model.merge_and_unload()
                merged_model.save_pretrained(save_path)
                print("Successfully merged and saved model with LoRA weights")
            except Exception as merge_error:
                print(f"Error merging LoRA weights: {merge_error}")
                print("Falling back to saving adapter only...")
                # Fallback: save just the LoRA adapter
                model.save_pretrained(save_path)
                print("Successfully saved LoRA adapter")
        else:
            # For full fine-tuning, save the entire model
            model.save_pretrained(save_path)
            print("Successfully saved full fine-tuned model")
        
        # Save the tokenizer
        tokenizer.save_pretrained(save_path)
        print(f"Tokenizer saved to {save_path}")
        
        return True
    except Exception as e:
        print(f"Error saving model: {e}")
        traceback.print_exc()
        return False


def finetune_model(
    dataset_name, model_name, epochs=10, batch_size=32, learning_rate=2e-5, use_lora=True, 
    lora_r=16, lora_alpha=32, lora_dropout=0.1
):
    print(f"Fine-tuning {model_name} on {dataset_name} dataset with {'LoRA' if use_lora else 'full parameters'}...")

    # Check if CUDA is available and print GPU info
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        # Optimize memory usage
        torch.cuda.empty_cache()
    else:
        print("No GPU available, using CPU")

    # Load the dataset
    if dataset_name == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
    elif dataset_name == "MR":
        X_train, y_train, X_test, y_test = load_MR()
    else:
        raise ValueError("Invalid dataset")

    # Encode labels
    le = LabelEncoder()
    le.fit(list(set(y_train)))
    y_train_encoded = le.transform(y_train)
    y_test_encoded = le.transform(y_test)
    n_classes = len(list(le.classes_))
    print(f"Classes: {le.classes_}, Total: {n_classes}")

    # Prepare datasets
    train_set = prepare_dataset(X_train, y_train_encoded)
    test_set = prepare_dataset(X_test, y_test_encoded)

    # Define tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare output directory for this run
    output_dir = f"./output/{dataset_name}_{model_name.replace('/', '_')}_{'lora' if use_lora else 'full'}"
    if os.path.exists(output_dir):
        print(f"Cleaning previous output directory: {output_dir}")
        try:
            shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        except Exception as e:
            print(f"Error cleaning directory: {e}")
            # If we can't clean it, use a timestamp to make it unique
            import time
            output_dir = f"{output_dir}_{int(time.time())}"
            os.makedirs(output_dir, exist_ok=True)

    # Load model with the correct number of output classes
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=n_classes,
        problem_type="single_label_classification",
        ignore_mismatched_sizes=True,  # This is crucial for handling mismatched classification heads
    )
    
    # Special handling for known problematic models
    if "siebert/sentiment-roberta-large-english" in model_name and use_lora:
        print(f"Using specialized LoRA config for {model_name}")
        # Use more conservative LoRA parameters for this model
        lora_r = 32
        lora_alpha = 64
        lora_dropout = 0.1
    
    # Apply LoRA if requested
    if use_lora:
        # Get the appropriate target modules for this model
        target_modules = get_model_specific_modules(model_name, model)
        print(f"Using target modules for LoRA: {target_modules}")
        
        # Define LoRA configuration
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # Sequence classification
            inference_mode=False,
            r=lora_r,  # Low-rank dimension
            lora_alpha=lora_alpha,  # Scale factor
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",  # Don't update bias terms for more stability
        )
        
        try:
            # Apply LoRA to the model
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()  # Show how many parameters are being trained
        except ValueError as e:
            print(f"Error applying LoRA: {e}")
            print("Trying alternate modules...")
            
            # Try some alternative target modules
            if "distilbert" in model_name.lower():
                alternate_modules = ["linear", "lin", "classifier", "dense"]
            elif "roberta" in model_name.lower():
                alternate_modules = ["out_proj", "dense.out", "self.out", "attention.output", "classifier"]
            else:
                alternate_modules = ["classifier", "dense", "out_proj", "linear"]
            
            # Try each alternate module set
            success = False
            for module in alternate_modules:
                try:
                    peft_config = LoraConfig(
                        task_type=TaskType.SEQ_CLS,
                        inference_mode=False,
                        r=lora_r,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                        target_modules=[module],
                        bias="none",  # Don't update bias terms for more stability
                    )
                    model = get_peft_model(model, peft_config)
                    print(f"Successfully applied LoRA with target module: {module}")
                    model.print_trainable_parameters()
                    success = True
                    break
                except ValueError:
                    continue
            
            # If all alternatives fail, fall back to full fine-tuning
            if not success:
                print("All LoRA attempts failed. Falling back to full fine-tuning")
                use_lora = False
    
    if not use_lora:
        print(f"Training all parameters. Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters")

    # Define tokenization function
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512)

    # Tokenize datasets
    tokenized_train_set = train_set.map(tokenize_function, batched=True)
    tokenized_test_set = test_set.map(tokenize_function, batched=True)

    # Use data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Use a unique run name
    run_name = f"{dataset_name}_{model_name.split('/')[-1]}_{'LoRA' if use_lora else 'Full'}_{torch.randint(0, 10000, (1,)).item()}"

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        eval_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=1e-4,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"./logs/{dataset_name}_{model_name.replace('/', '_')}",
        logging_steps=100,
        push_to_hub=False,
        fp16=False,  # Disable mixed precision to avoid FP16 gradient errors
        dataloader_num_workers=0,  # Avoid multiprocessing issues on Windows
        report_to="wandb" if os.environ.get("WANDB_DISABLED") != "true" else [],
        # LoRA-specific settings for better efficiency
        gradient_accumulation_steps=4 if use_lora else 1,  # Reduce memory usage
        group_by_length=True,  # Group similar length sequences for efficiency
        save_total_limit=2,  # Only keep the 2 most recent checkpoints
    )

    # Set up trainer
    class SafeTrainer(Trainer):
        """Custom trainer that safely loads the best model without crashing"""
        def _load_best_model(self):
            try:
                # Try to load the best model checkpointed during training
                super()._load_best_model()
                print("Successfully loaded best model")
            except Exception as e:
                print(f"Error loading best model: {e}")
                print("Continuing with current model state")
                # No need to try to load again
    
    # Add custom callback to handle end of training
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=2),
        SkipBestModelLoadingCallback()
    ]
    
    trainer = SafeTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_set,
        eval_dataset=tokenized_test_set,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
    )

    # Train the model with error handling
    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        # If we had a training error, we'll still try to evaluate with the current model
        print("Attempting to continue with evaluation despite training error...")

    # Save the model immediately after training completes, before evaluation
    print("Saving model after training...")
    model_save_path = f"./saved_models/{dataset_name}_{model_name.replace('/', '_')}_{'lora' if use_lora else 'full'}"
    save_success = safe_save_model(model, tokenizer, model_save_path, use_lora)
    
    if not save_success:
        print("Warning: Failed to save model using safe_save_model function")
        # Try the traditional approach as fallback
        try:
            os.makedirs(model_save_path, exist_ok=True)
            if use_lora:
                model.save_pretrained(model_save_path)
            else:
                trainer.save_model(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f"Model saved using fallback method to {model_save_path}")
        except Exception as e:
            print(f"Error during fallback model saving: {e}")
    
    # Evaluate on test set
    try:
        eval_results = trainer.evaluate(tokenized_test_set)
        print(f"Test set evaluation results: {eval_results}")
    except Exception as e:
        print(f"Error during evaluation: {e}")
        eval_results = {"error": str(e)}

    return eval_results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fine-tune models for sentiment analysis with LoRA"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["MR", "Semeval2017A"],
        default="MR",
        help="Dataset to use (MR or Semeval2017A)",
    )
    parser.add_argument("--model", type=str, help="Specific model to fine-tune")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument(
        "--learning_rate", type=float, default=2e-5, help="Learning rate"
    )
    parser.add_argument(
        "--no_wandb", action="store_true", help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--no_lora", action="store_true", help="Disable LoRA (use full fine-tuning)"
    )
    parser.add_argument(
        "--lora_r", type=int, default=16, help="LoRA rank parameter"
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.1, help="LoRA dropout parameter"
    )
    args = parser.parse_args()

    # Disable wandb if requested
    if args.no_wandb:
        os.environ["WANDB_DISABLED"] = "true"

    # Models to fine-tune for each dataset - ensure they're suitable for the task
    mr_models = [
        "siebert/sentiment-roberta-large-english",
        "clapAI/modernBERT-large-multilingual-sentiment",
        "Elron/deberta-v3-large-sentiment",
    ]

    semeval_models = [
        "agentlans/deberta-v3-base-tweet-sentiment",
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "siebert/sentiment-roberta-large-english",
    ]

    DATASET = args.dataset
    USE_LORA = not args.no_lora
    #USE_LORA = False

    if args.model:
        # Fine-tune a specific model
        results = finetune_model(
            DATASET, args.model, epochs=args.epochs, batch_size=args.batch_size, 
            learning_rate=args.learning_rate, use_lora=USE_LORA, 
            lora_r=args.lora_r, lora_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout
        )
        print(f"\nResults for {args.model} on {DATASET} with {'LoRA' if USE_LORA else 'full parameters'}:")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"{metric}: {value:.4f}")
            else:
                print(f"{metric}: {value}")
    else:
        # Use the appropriate model list based on dataset
        models_to_use = mr_models if DATASET == "MR" else semeval_models

        # Store results for comparison
        results = {}

        # Fine-tune and evaluate each model
        for model_name in models_to_use:
            print(f"\n{'='*50}\nFine-tuning {model_name} on {DATASET} with {'LoRA' if USE_LORA else 'full parameters'}\n{'='*50}")
            try:
                model_results = finetune_model(
                    DATASET, model_name, epochs=args.epochs, batch_size=args.batch_size,
                    learning_rate=args.learning_rate, use_lora=USE_LORA, 
                    lora_r=args.lora_r, lora_alpha=args.lora_alpha, 
                    lora_dropout=args.lora_dropout
                )
                results[model_name] = model_results
            except Exception as e:
                print(f"Error fine-tuning {model_name}: {e}")
                import traceback

                traceback.print_exc()
                results[model_name] = {"error": str(e)}

        # Print comparative results
        print("\n\nResults Summary for", DATASET, "with", "LoRA" if USE_LORA else "full parameters")
        print("-" * 80)
        print(f"{'Model':<40} {'Accuracy':<10} {'F1 Score':<10} {'Recall':<10}")
        print("-" * 80)

        for model_name, metrics in results.items():
            model_short = model_name.split("/")[-1]
            if "error" in metrics:
                print(f"{model_short:<40} {'ERROR':<10} {'ERROR':<10} {'ERROR':<10}")
            else:
                print(
                    f"{model_short:<40} {metrics.get('eval_accuracy', 0):<10.4f} {metrics.get('eval_f1', 0):<10.4f} {metrics.get('eval_recall', 0):<10.4f}"
                )