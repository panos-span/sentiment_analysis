from transformers import pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from utils.load_datasets import load_MR, load_Semeval2017A
from training import get_metrics_report
import argparse

# DATASET = 'MR'
# PRETRAINED_MODEL = 'siebert/sentiment-roberta-large-english'

DATASET = 'Semeval2017A'
PRETRAINED_MODEL = 'cardiffnlp/twitter-roberta-base-sentiment'


#LABELS_MAPPING = {
#    'siebert/sentiment-roberta-large-english': {
#        'POSITIVE': 'positive',
#        'NEGATIVE': 'negative',
#    },
#    'cardiffnlp/twitter-roberta-base-sentiment': {
#        'LABEL_0': 'negative',
#        'LABEL_1': 'neutral',
#        'LABEL_2': 'positive',
#    }
#}

LABELS_MAPPING = {
    'siebert/sentiment-roberta-large-english': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'cardiffnlp/twitter-roberta-base-sentiment': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    },
    'clapAI/modernBERT-large-multilingual-sentiment': {
        'positive': 'positive',
        'negative': 'negative', 
        'neutral': 'neutral'
    },
    'jbeno/electra-large-classifier-sentiment': {
        'POSITIVE': 'positive',
        'NEGATIVE': 'negative',
    },
    'jayanta/google-electra-base-discriminator-english-sentweet-sentiment': {
        'LABEL_0': 'negative',
        'LABEL_1': 'neutral',
        'LABEL_2': 'positive',
    }
}

def evaluate_model(dataset, model_name):
    print(f"Evaluating {model_name} on {dataset} dataset...")
    
    # load the raw data
    if dataset == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
    elif dataset == "MR":
        X_train, y_train, X_test, y_test = load_MR()
    else:
        raise ValueError("Invalid dataset")

    # encode labels
    le = LabelEncoder()
    le.fit(list(set(y_train)))
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)
    print(f"Label mapping: {le.classes_}")      

    # define a proper pipeline
    sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

    y_pred = []
    for x in tqdm(X_test):
        # Get the prediction from the pipeline
        result = sentiment_pipeline(x)[0]  # Get the first (and only) result
        label = result['label']  # Extract the predicted label
        
        # Map the model's label format to our dataset format
        if label in LABELS_MAPPING[model_name]:
            mapped_label = LABELS_MAPPING[model_name][label]
            y_pred.append(mapped_label)
        else:
            print(f"Warning: Unknown label {label} from model {model_name}")
            # Use the most common label as a fallback
            y_pred.append(le.classes_[0])  

    y_pred = le.transform(y_pred)
    report = get_metrics_report(y_test, y_pred)
    print(f'\nDataset: {dataset}\nPre-Trained model: {model_name}\nTest set evaluation\n{report}')
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate pre-trained models for sentiment analysis')
    parser.add_argument('--dataset', type=str, choices=['MR', 'Semeval2017A'], default='MR',
                        help='Dataset to use (MR or Semeval2017A)')
    parser.add_argument('--all', action='store_true', help='Evaluate all models')
    args = parser.parse_args()
    
    if args.dataset == 'MR':
        models = [
            'siebert/sentiment-roberta-large-english',
            'clapAI/modernBERT-large-multilingual-sentiment',
            'jbeno/electra-large-classifier-sentiment'
        ]
    else:  # Semeval2017A
        models = [
            'cardiffnlp/twitter-roberta-base-sentiment',
            'clapAI/modernBERT-large-multilingual-sentiment',
            'jayanta/google-electra-base-discriminator-english-sentweet-sentiment'
        ]
    
    if args.all:
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
            # Extract metrics from report string
            lines = report.strip().split('\n')
            accuracy = lines[0].split(': ')[1]
            recall = lines[1].split(': ')[1]
            f1 = lines[2].split(': ')[1]
            model_short = model.split('/')[-1]
            print(f"{model_short} | {accuracy} | {recall} | {f1}")
    else:
        # Default behavior: evaluate first model in the list
        evaluate_model(args.dataset, models[0])