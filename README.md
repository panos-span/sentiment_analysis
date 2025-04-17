# Sentiment Analysis Using Advanced NLP Models

This repository contains the implementation of various deep learning architectures for sentiment analysis, developed as part of the Natural Language Processing course at the National Technical University of Athens.

## Project Overview

The project explores different approaches to sentiment analysis, moving from simple architectures to more complex ones:

1. **Enhanced Word Embedding Pooling**: Combining mean and max pooling of word embeddings
2. **LSTM Networks**: Using LSTMs (including bidirectional variants) for sequential text processing
3. **Self-Attention Mechanisms**: Implementing single-head attention for text classification
4. **Multi-Head Attention**: Extending to multiple attention heads for improved performance
5. **Transformer Encoders**: Building full transformer encoder architecture
6. **Pre-Trained Models**: Leveraging state-of-the-art transformer models
7. **Fine-Tuning**: Parameter-efficient fine-tuning of pre-trained models

## Datasets

The project uses two datasets for sentiment classification:
- **Movie Review (MR)**: A binary sentiment classification dataset of movie reviews
- **Semeval2017A**: A Twitter dataset with three sentiment classes (positive, negative, neutral)

## Prerequisites

The project requires Python 3 and the following packages:
```
numpy==1.24.3
tqdm==4.65.0
torch==1.13.1
scikit_learn==1.2.2
nltk==3.8.1
transformers==4.29.2
datasets==2.12.0
accelerate==0.19.0
evaluate==0.4.0
```

## Installation

1. Create a virtual environment (recommended):
```bash
conda create -n sentiment_analysis python=3.8
conda activate sentiment_analysis
```

2. Install PyTorch following the instructions from the [PyTorch website](https://pytorch.org/)

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download pre-trained word embeddings:
   - [Glove 6B](http://nlp.stanford.edu/data/glove.6B.zip): Generic English word embeddings
   - [Glove Twitter](http://nlp.stanford.edu/data/glove.twitter.27B.zip): Twitter-specific word embeddings
   - [fastText](https://fasttext.cc/docs/en/english-vectors.html): Generic English word embeddings

   Place the downloaded embeddings in the `/embeddings` folder.

## Project Structure

```
├── attention.py          # Attention-based model implementations
├── config.py             # Project configuration
├── dataloading.py        # Dataset loading utilities
├── early_stopper.py      # Early stopping implementation
├── finetune_pretrained.py # Fine-tuning for pre-trained models
├── main.py               # Main script to run experiments
├── models.py             # DNN and LSTM model implementations
├── openai_eval.py        # ChatGPT evaluation
├── transfer_pretrained.py # Evaluation of pre-trained models
├── training.py           # Training utilities
├── utils/                # Utility functions
│   ├── load_datasets.py  # Functions to load datasets
│   └── load_embeddings.py # Functions to load word embeddings
├── embeddings/           # Directory for storing embeddings
└── datasets/             # Directory for storing datasets
    ├── MR/               # Movie Review dataset
    └── Semeval2017A/     # Semeval2017A dataset
```

## Usage

### Running Experiments with Custom Models

To train and evaluate custom models:

```bash
python main.py
```

You can modify the configuration in `main.py` to select:
- The dataset (`DATASET` variable)
- The model architecture (uncomment the desired model)
- Embedding dimensions and file path
- Training parameters (batch size, epochs, etc.)

### Using Pre-trained Transformers

To evaluate pre-trained transformer models:

```bash
python transfer_pretrained.py --dataset MR --all
```

Replace `MR` with `Semeval2017A` for evaluating on the Twitter dataset.

### Fine-tuning Pre-trained Transformers

To fine-tune pre-trained transformer models:

```bash
python finetune_pretrained.py
```

For better performance, it's recommended to run fine-tuning on Google Colab with GPU acceleration.

## Model Architectures

### 1. Enhanced Pooling
Combines mean and max pooling of word embeddings for improved feature representation:
```
u = [mean(E) || max(E)]
```

### 2. LSTM Networks
Processes text sequentially, capturing contextual information and long-distance dependencies.

### 3. Self-Attention
Dynamically focuses on relevant words in the text using a single attention head.

### 4. Multi-Head Attention
Extends self-attention with multiple heads to capture different aspects of the text.

### 5. Transformer Encoder
Implements the encoder component of the transformer architecture with self-attention, layer normalization, and feed-forward networks.

### 6. Pre-trained Models
Leverages existing transformer models that have been pre-trained on large corpora.

### 7. Fine-tuned Models
Fine-tunes pre-trained transformers using parameter-efficient techniques.

## Results

Our experiments show a clear progression in performance:

1. Custom models achieve reasonable performance, with bidirectional LSTMs reaching ~77% accuracy on the MR dataset.
2. Transformer-based architectures show comparable performance to LSTMs with more flexibility.
3. Pre-trained transformer models significantly outperform custom models, with the best achieving 92.60% accuracy on MR.
4. Fine-tuning further improves performance, especially for models that initially perform poorly in zero-shot settings.

## License

This project is available under the MIT License.

## Acknowledgments

- National Technical University of Athens, Natural Language Processing course
- Hugging Face for the transformers library
- OpenAI for ChatGPT evaluations
