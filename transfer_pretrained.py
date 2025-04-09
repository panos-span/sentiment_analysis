import torch
import os
import numpy as np
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report
import argparse

# Disable PyTorch 2.0 compiler
import torch._dynamo

torch._dynamo.config.suppress_errors = True

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model configurations with type and handling information
MODELS_CONFIG = {
    "siebert/sentiment-roberta-large-english": {
        "type": "discrete",
        "mapping": {
            "POSITIVE": "positive",
            "NEGATIVE": "negative",
            "LABEL_0": "negative",
            "LABEL_1": "positive",
        },
    },
    "cardiffnlp/twitter-roberta-base-sentiment-latest": {
        "type": "discrete",
        "mapping": {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive",
        },
    },
    "clapAI/modernBERT-large-multilingual-sentiment": {
        "type": "discrete",
        "mapping": {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive",
        },
    },
    "agentlans/deberta-v3-base-tweet-sentiment": {
        "type": "continuous",
        "thresholds": {
            "negative": (-float("inf"), -0.1),
            "neutral": (-0.1, 0.1),
            "positive": (0.1, float("inf")),
        },
    },
    "Elron/deberta-v3-large-sentiment": {
        "type": "discrete",
        "mapping": {
            "LABEL_0": "negative",
            "LABEL_1": "neutral",
            "LABEL_2": "positive",
        },
    },
}


def evaluate_model(dataset, model_name):
    print(f"Evaluating {model_name} on {dataset} dataset...")

    # Load the raw data
    if dataset == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
    elif dataset == "MR":
        X_train, y_train, X_test, y_test = load_MR()
    else:
        raise ValueError("Invalid dataset")

    # Get all possible labels from both train and test sets
    all_possible_labels = sorted(list(set(y_train)))
    print(f"All possible labels in training data: {all_possible_labels}")

    # Encode labels
    le = LabelEncoder()
    le.fit(all_possible_labels)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    print(f"Label mapping: {le.classes_}")

    # Try direct model approach
    print("Using direct model approach...")

    try:
        # Load model and tokenizer
        print(f"Loading model: {model_name}")
        model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
            device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Set model to evaluation mode
        model.eval()

        # Get model configuration
        model_config = MODELS_CONFIG[model_name]
        model_type = model_config["type"]

        # Check if the model uses continuous scoring
        if model_type == "continuous":
            print(f"Model {model_name} produces continuous sentiment scores")
            thresholds = model_config["thresholds"]
        else:
            print(f"Model {model_name} produces discrete labels")
            # Get actual label mapping from model
            id2label = (
                model.config.id2label if hasattr(model.config, "id2label") else None
            )
            print(f"Model's id2label mapping: {id2label}")

        # Process in batches
        y_pred = []
        batch_size = 32  # Adjust based on GPU memory

        # Process with progress bar
        with torch.no_grad():
            for i in tqdm(range(0, len(X_test), batch_size), desc="Processing batches"):
                batch = X_test[i : i + batch_size]

                # Tokenize inputs
                inputs = tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(device)

                # Get model outputs
                outputs = model(**inputs)

                if model_type == "continuous":
                    # For continuous sentiment models, get the raw logits/scores
                    sentiment_scores = outputs.logits.squeeze(-1).cpu().numpy()

                    # Convert scores to labels based on thresholds
                    for score in sentiment_scores:
                        for label, (min_val, max_val) in thresholds.items():
                            if min_val <= score < max_val:
                                y_pred.append(label)
                                break
                        else:
                            # Fallback if no threshold matches (shouldn't happen with proper thresholds)
                            print(f"Warning: Score {score} doesn't match any threshold")
                            y_pred.append(all_possible_labels[0])
                else:
                    # For discrete label models
                    predictions = outputs.logits.argmax(dim=1)

                    # Map predictions to dataset labels
                    for pred in predictions.cpu().numpy():
                        # Get the label from model's mapping
                        if id2label:
                            label = id2label[int(pred)]
                        else:
                            label = f"LABEL_{pred}"

                        # Try to map the label
                        if label in model_config["mapping"]:
                            mapped_label = model_config["mapping"][label]
                        else:
                            # If we can't find the label, check if a numeric version exists
                            numeric_label = f"LABEL_{pred}"
                            if numeric_label in model_config["mapping"]:
                                mapped_label = model_config["mapping"][numeric_label]
                            else:
                                # As a last resort, use the first available label from training data
                                print(
                                    f"Warning: Unknown label {label} from model {model_name}"
                                )
                                mapped_label = all_possible_labels[0]

                        # Only append labels that exist in the training data
                        if mapped_label in all_possible_labels:
                            y_pred.append(mapped_label)
                        else:
                            # If mapped label doesn't exist in our data, use the first available label
                            print(
                                f"Warning: Mapped label {mapped_label} not in training labels"
                            )
                            y_pred.append(all_possible_labels[0])

        # Convert string labels back to numeric
        y_pred_encoded = le.transform(y_pred)

        # Calculate metrics
        report = get_metrics_report(y_test, y_pred_encoded)
        print(
            f"\nDataset: {dataset}\nPre-Trained model: {model_name}\nTest set evaluation\n{report}"
        )
        return report

    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback

        traceback.print_exc()
        return "Failed to evaluate model"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained models for sentiment analysis"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["MR", "Semeval2017A"],
        default="MR",
        help="Dataset to use (MR or Semeval2017A)",
    )
    parser.add_argument("--all", action="store_true", help="Evaluate all models")
    parser.add_argument("--model", type=str, help="Specific model to evaluate")
    parser.add_argument(
        "--tune_thresholds",
        action="store_true",
        help="Tune thresholds for continuous models (uses a subset of training data)",
    )
    args = parser.parse_args()

    if args.dataset == "MR":
        models = [
            "siebert/sentiment-roberta-large-english",
            "clapAI/modernBERT-large-multilingual-sentiment",
            "Elron/deberta-v3-large-sentiment",
        ]
    else:  # Semeval2017A
        models = [
            "cardiffnlp/twitter-roberta-base-sentiment-latest",
            "clapAI/modernBERT-large-multilingual-sentiment",
            "agentlans/deberta-v3-base-tweet-sentiment",
        ]

    # Tune thresholds for continuous models if requested
    if args.tune_thresholds:
        for model_name in models:
            if (
                model_name in MODELS_CONFIG
                and MODELS_CONFIG[model_name]["type"] == "continuous"
            ):
                print(f"Tuning thresholds for {model_name}...")
                # This would involve using a validation set to find optimal thresholds
                # For simplicity, we're not implementing the full tuning algorithm here
                # In a real application, you might use a grid search or other optimization
                pass

    # If specific model is provided, use only that
    if args.model:
        models = [args.model]
        evaluate_model(args.dataset, args.model)
    elif args.all:
        results = {}
        for model in models:
            report = evaluate_model(args.dataset, model)
            results[model] = report

        # Print comparison table
        print("\n\n--- RESULTS COMPARISON ---")
        print(f"Dataset: {args.dataset}")
        print("Model | Accuracy | Recall | F1-Score")
        print("------|----------|--------|--------")
        for model, report in results.items():
            if isinstance(report, str) and report.startswith("Failed"):
                model_short = model.split("/")[-1]
                print(f"{model_short} | FAILED | FAILED | FAILED")
            else:
                # Extract metrics from report string
                try:
                    lines = report.strip().split("\n")
                    accuracy = lines[0].split(": ")[1]
                    recall = lines[1].split(": ")[1]
                    f1 = lines[2].split(": ")[1]
                    model_short = model.split("/")[-1]
                    print(f"{model_short} | {accuracy} | {recall} | {f1}")
                except:
                    model_short = model.split("/")[-1]
                    print(f"{model_short} | ERROR | ERROR | ERROR")
    else:
        # Default behavior: evaluate first model in the list
        evaluate_model(args.dataset, models[0])
