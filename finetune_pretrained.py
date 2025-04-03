import numpy as np
import evaluate
from datasets import Dataset
from transformers import (
    TrainingArguments, 
    Trainer, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    EarlyStoppingCallback
)
from sklearn.preprocessing import LabelEncoder
from utils.load_datasets import load_MR, load_Semeval2017A

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
    recall = recall_metric.compute(predictions=predictions, references=labels, average="macro")
    
    return {
        'accuracy': accuracy['accuracy'],
        'f1': f1['f1'],
        'recall': recall['recall']
    }

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

def prepare_dataset(X, y):
    texts, labels = [], []
    for text, label in zip(X, y):
        texts.append(text)
        labels.append(label)
    return Dataset.from_dict({'text': texts, 'label': labels})

def finetune_model(dataset_name, model_name, epochs=5, batch_size=16, learning_rate=2e-5):
    print(f"Fine-tuning {model_name} on {dataset_name} dataset...")
    
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
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=n_classes
    )
    
    # Create a tokenize function with the specific tokenizer
    def tokenize_with_tokenizer(examples):
        return tokenize_function(examples, tokenizer)
    
    # Tokenize datasets
    tokenized_train_set = train_set.map(tokenize_with_tokenizer, batched=True)
    tokenized_test_set = test_set.map(tokenize_with_tokenizer, batched=True)
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=f"./output/{dataset_name}_{model_name.replace('/', '_')}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size*2,
        learning_rate=learning_rate,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=f"./logs/{dataset_name}_{model_name.replace('/', '_')}",
        logging_steps=50,
        push_to_hub=False,
    )
    
    # Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_set,
        eval_dataset=tokenized_test_set,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate on test set
    eval_results = trainer.evaluate(tokenized_test_set)
    print(f"Test set evaluation results: {eval_results}")
    
    # Save the model
    model_save_path = f"./saved_models/{dataset_name}_{model_name.replace('/', '_')}"
    trainer.save_model(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    return eval_results

if __name__ == '__main__':
    # Models to fine-tune for each dataset
    mr_models = [
        'siebert/sentiment-roberta-large-english',
        'clapAI/modernBERT-large-multilingual-sentiment',
        'distilbert-base-uncased'
    ]
    
    semeval_models = [
        'jbeno/electra-large-classifier-sentiment',
        'clapAI/modernBERT-large-multilingual-sentiment',
        'jayanta/google-electra-base-discriminator-english-sentweet-sentiment'
    ]
    
    # Uncomment the dataset you want to use
    DATASET = 'MR'
    # DATASET = 'Semeval2017A'
    
    if DATASET == 'MR':
        models_to_use = mr_models
    else:
        models_to_use = semeval_models
    
    # Store results for comparison
    results = {}
    
    # Fine-tune and evaluate each model
    for model_name in models_to_use:
        print(f"\n{'='*50}\nFine-tuning {model_name} on {DATASET}\n{'='*50}")
        model_results = finetune_model(DATASET, model_name, epochs=5)
        results[model_name] = model_results
    
    # Print comparative results
    print("\n\nResults Summary for", DATASET)
    print("-" * 80)
    print(f"{'Model':<30} {'Accuracy':<10} {'F1 Score':<10} {'Recall':<10}")
    print("-" * 80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<30} {metrics['eval_accuracy']:<10.4f} {metrics['eval_f1']:<10.4f} {metrics['eval_recall']:<10.4f}")