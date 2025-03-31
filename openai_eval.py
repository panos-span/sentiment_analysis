import random
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import openai
import time
import json
from utils.load_datasets import load_MR, load_Semeval2017A

class ChatGPTSentimentAnalyzer:
    def __init__(self, api_key, model="gpt-4o-mini"):
        """
        Initialize the ChatGPT sentiment analyzer.
        
        Args:
            api_key (str): OpenAI API key
            model (str): Model to use
        """
        openai.api_key = api_key
        self.model = model
        
        # Define different prompt templates to experiment with
        self.prompts = {
            "basic": "Please classify the sentiment of the following text as positive, negative, or neutral:\n\n\"{text}\"",
            
            "few_shot": """I'm going to give you a few examples of text with their sentiment, and then ask you to classify a new text.

Positive example: "I love this product, it's amazing and works perfectly!"
Negative example: "Horrible experience, would never recommend this to anyone."
Neutral example: "The package arrived yesterday as scheduled."

Now, please classify the sentiment of this text as EXACTLY one of: positive, negative, or neutral.
Text: "{text}"
Classification:""",
            
            "reasoning": """Please analyze the sentiment of the following text:
"{text}"

1. First, classify the sentiment as EXACTLY one of: positive, negative, or neutral.
2. Then, explain your reasoning in detail.
3. Specifically highlight the 3-5 most important words or phrases that influenced your classification.
4. Rate how confident you are in your classification on a scale of 1-10.

Format your response as follows:
Classification: [your classification]
Confidence: [number 1-10]
Reasoning: [your explanation]
Key words/phrases: [list the important words]""",
            
            "movie_context": """You are analyzing movie reviews to determine if they express positive or negative sentiment.
Consider aspects like plot, acting, direction, and overall enjoyment.

Review: "{text}"

Please classify this movie review as EXACTLY one of:
- Positive: The reviewer enjoyed the movie
- Negative: The reviewer did not enjoy the movie""",
            
            "twitter_context": """You are analyzing tweets to determine their sentiment. Tweets often use informal language, hashtags, and emoticons.

Tweet: "{text}"

Please classify this tweet's sentiment as EXACTLY one of:
- Positive: The tweet expresses positive emotions or approval
- Negative: The tweet expresses negative emotions or criticism
- Neutral: The tweet is factual, objective, or doesn't express a clear sentiment"""
        }
    
    def get_sentiment(self, text, prompt_type="basic"):
        """
        Get sentiment classification from ChatGPT.
        
        Args:
            text (str): Text to classify
            prompt_type (str): Type of prompt to use
            
        Returns:
            dict: Response containing sentiment classification and explanation
        """
        prompt = self.prompts.get(prompt_type, self.prompts["basic"])
        prompt = prompt.format(text=text)
        
        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that analyzes sentiment."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1  # Lower temperature for more consistent results
            )
            
            return {
                "response_text": response.choices[0].message.content.strip(),
                "prompt_type": prompt_type,
                "text": text
            }
        except Exception as e:
            print(f"Error calling OpenAI API: {e}")
            return {"error": str(e), "text": text, "prompt_type": prompt_type}
        
    def parse_sentiment(self, response):
        """
        Parse the ChatGPT response to extract sentiment classification.
        
        Args:
            response (dict): Response from get_sentiment method
            
        Returns:
            str: Extracted sentiment (positive, negative, or neutral)
        """
        if "error" in response:
            return "error"
        
        response_text = response["response_text"].lower()
        prompt_type = response["prompt_type"]
        
        # Different parsing strategies based on prompt type
        if prompt_type == "reasoning":
            # Try to extract from structured format
            if "classification:" in response_text:
                classification_line = [line for line in response_text.split('\n') 
                                      if line.startswith("classification:")][0]
                sentiment = classification_line.split(":", 1)[1].strip().lower()
            else:
                # Fallback to keyword detection
                if "positive" in response_text[:100]:
                    sentiment = "positive"
                elif "negative" in response_text[:100]:
                    sentiment = "negative"
                else:
                    sentiment = "neutral"
        else:
            # For other prompt types, look for sentiment keywords
            if "positive" in response_text[:100]:
                sentiment = "positive"
            elif "negative" in response_text[:100]:
                sentiment = "negative"
            else:
                sentiment = "neutral"
        
        # Normalize to match dataset labels
        if sentiment in ["positive", "positive:"]:
            return "positive"
        elif sentiment in ["negative", "negative:"]:
            return "negative"
        elif sentiment in ["neutral", "neutral:"]:
            return "neutral"
        else:
            return "unknown"
            
    def extract_important_words(self, response):
        """
        Extract the important words identified in the reasoning prompt response.
        
        Args:
            response (dict): Response from get_sentiment method
            
        Returns:
            list: Important words/phrases identified
        """
        if "error" in response or response["prompt_type"] != "reasoning":
            return []
        
        response_text = response["response_text"]
        
        # Try to extract the key words section
        if "key words/phrases:" in response_text.lower():
            key_words_section = response_text.lower().split("key words/phrases:")[1].strip()
            # Extract words (this is a simple implementation, could be improved)
            words = [w.strip() for w in key_words_section.split(',')]
            # Clean up any bullet points or numbering
            words = [w.strip('- "\'•').strip() for w in words]
            return [w for w in words if w]
        
        return []

def get_dataset_samples(dataset_name, n_samples_per_class=20, random_seed=42):
    """
    Get samples from dataset, ensuring equal representation from each class.
    
    Args:
        dataset_name (str): Name of dataset ("MR" or "Semeval2017A")
        n_samples_per_class (int): Number of samples per class
        random_seed (int): Random seed for reproducibility
        
    Returns:
        tuple: (samples, classes) where samples is a list of (text, label) tuples
    """
    random.seed(random_seed)
    
    if dataset_name == "MR":
        X_train, y_train, X_test, y_test = load_MR()
        # Combine train and test for sampling
        X = X_train + X_test
        y = y_train + y_test
    elif dataset_name == "Semeval2017A":
        X_train, y_train, X_test, y_test = load_Semeval2017A()
        X = X_train + X_test
        y = y_train + y_test
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    classes = sorted(list(set(y)))
    samples = []
    
    for cls in classes:
        # Get indices for this class
        indices = [i for i, label in enumerate(y) if label == cls]
        # Sample n_samples_per_class random indices
        sampled_indices = random.sample(indices, min(n_samples_per_class, len(indices)))
        # Get the samples
        for idx in sampled_indices:
            samples.append((X[idx], cls))
    
    return samples, classes

def evaluate_chatgpt(analyzer, dataset_name, prompt_type, n_samples_per_class=20):
    """
    Evaluate ChatGPT on a dataset using a specific prompt type.
    
    Args:
        analyzer (ChatGPTSentimentAnalyzer): Analyzer instance
        dataset_name (str): Name of dataset
        prompt_type (str): Type of prompt to use
        n_samples_per_class (int): Number of samples per class
        
    Returns:
        dict: Evaluation results
    """
    samples, classes = get_dataset_samples(dataset_name, n_samples_per_class)
    
    results = []
    important_words = {cls: [] for cls in classes}
    
    for i, (text, true_label) in enumerate(samples):
        print(f"Processing sample {i+1}/{len(samples)}...")
        
        # Get sentiment from ChatGPT
        response = analyzer.get_sentiment(text, prompt_type)
        predicted_label = analyzer.parse_sentiment(response)
        
        # Extract important words if using reasoning prompt
        if prompt_type == "reasoning":
            words = analyzer.extract_important_words(response)
            if words:
                important_words[true_label].extend(words)
        
        # Store the full response for analysis
        results.append({
            "text": text,
            "true_label": true_label,
            "predicted_label": predicted_label,
            "response": response
        })
        
        # Rate limit to avoid hitting API limits
        time.sleep(1)
    
    # Calculate metrics
    true_labels = [r["true_label"] for r in results]
    pred_labels = [r["predicted_label"] for r in results if r["predicted_label"] != "unknown"]
    
    # Filter out any unknowns for metric calculation
    filtered_true = [true_labels[i] for i, p in enumerate(pred_labels) if p != "unknown"]
    
    if len(filtered_true) > 0:
        accuracy = accuracy_score(filtered_true, pred_labels)
        f1 = f1_score(filtered_true, pred_labels, average='macro')
        recall = recall_score(filtered_true, pred_labels, average='macro')
        cm = confusion_matrix(filtered_true, pred_labels, labels=classes)
        
        # Convert confusion matrix to a list for easier serialization
        cm_list = cm.tolist()
    else:
        accuracy = f1 = recall = 0
        cm_list = []
    
    # Analyze important words (for reasoning prompt)
    word_frequency = {}
    for cls in important_words:
        word_counts = {}
        for word in important_words[cls]:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Sort by frequency
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        word_frequency[cls] = sorted_words
    
    # Compile all results
    evaluation = {
        "dataset": dataset_name,
        "prompt_type": prompt_type,
        "metrics": {
            "accuracy": accuracy,
            "f1_score": f1,
            "recall": recall
        },
        "confusion_matrix": {
            "matrix": cm_list,
            "labels": classes
        },
        "important_words": word_frequency,
        "results": results
    }
    
    return evaluation

# Example usage (comment out when running the main experiment)
# api_key = "your-openai-api-key"
# analyzer = ChatGPTSentimentAnalyzer(api_key)
# results_mr = evaluate_chatgpt(analyzer, "MR", "reasoning")
# results_semeval = evaluate_chatgpt(analyzer, "Semeval2017A", "reasoning")

def run_experiments(api_key, save_path="chatgpt_sentiment_results.json"):
    """
    Run experiments on both datasets with different prompt types.
    
    Args:
        api_key (str): OpenAI API key
        save_path (str): Path to save the results
        
    Returns:
        dict: All experiment results
    """
    analyzer = ChatGPTSentimentAnalyzer(api_key)
    
    experiments = {
        "MR": ["basic", "few_shot", "reasoning", "movie_context"],
        "Semeval2017A": ["basic", "few_shot", "reasoning", "twitter_context"]
    }
    
    all_results = {}
    
    for dataset in experiments:
        all_results[dataset] = {}
        for prompt_type in experiments[dataset]:
            print(f"\nEvaluating {dataset} with {prompt_type} prompt...")
            results = evaluate_chatgpt(analyzer, dataset, prompt_type)
            all_results[dataset][prompt_type] = results
            
            # Print summary after each experiment
            print(f"\nResults for {dataset} with {prompt_type} prompt:")
            print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
            print(f"F1 Score: {results['metrics']['f1_score']:.4f}")
            print(f"Recall: {results['metrics']['recall']:.4f}")
            
            # Save results incrementally
            with open(save_path, 'w') as f:
                json.dump(all_results, f, indent=2)
    
    print(f"\nAll experiments completed and saved to {save_path}")
    return all_results

def analyze_errors(results):
    """
    Analyze the classification errors made by ChatGPT.
    
    Args:
        results (dict): Results from evaluate_chatgpt
        
    Returns:
        dict: Error analysis
    """
    errors = []
    
    for sample in results["results"]:
        if sample["predicted_label"] != sample["true_label"] and sample["predicted_label"] != "unknown":
            errors.append({
                "text": sample["text"],
                "true_label": sample["true_label"],
                "predicted_label": sample["predicted_label"],
                "response": sample["response"]["response_text"]
            })
    
    return {
        "total_samples": len(results["results"]),
        "error_count": len(errors),
        "error_rate": len(errors) / len(results["results"]),
        "errors": errors
    }

# Note: To run the experiments, you would need to call:
# run_experiments("your-openai-api-key")