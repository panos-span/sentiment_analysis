from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from multiprocessing import Pool, cpu_count
# Try to use ekphrasis for Twitter data
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons

class SentenceDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """

    def __init__(self, X, y, word2idx):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples
        
        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """
        # Efficiently check for required resources
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        # Store the inputs
        self.labels = y
        self.word2idx = word2idx
        
        # Check if we're dealing with Twitter data with minimal scanning
        # Sample just a subset of data for faster detection
        sample_size = min(100, len(X))
        is_twitter_data = any('@' in text or '#' in text or 'http' in text for text in X[:sample_size])
        
        # Pre-allocate data list with appropriate size for better memory management
        self.data = [None] * len(X)
        
        if is_twitter_data:
            try:   
                # Initialize the processor once outside the loop
                text_processor = TextPreProcessor(
                    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                            'time', 'date', 'number'],
                    annotate={"hashtag", "allcaps", "elongated", "repeated",
                            'emphasis', 'censored'},
                    fix_html=True,
                    segmenter="twitter", 
                    corrector="twitter",
                    unpack_hashtags=True,
                    unpack_contractions=True,
                    spell_correct_elong=False,
                    tokenizer=SocialTokenizer(lowercase=True).tokenize,
                    dicts=[emoticons]
                )
                
                # Use tqdm for progress tracking
                for i, text in enumerate(tqdm(X, desc="Tokenizing tweets")):
                    self.data[i] = text_processor.pre_process_doc(text)
                    
            except ImportError:
                raise ImportError("Ekphrasis is not installed. Please install it to process Twitter data.")
        else:
            # For standard tokenization, use batch processing where possible
            # Define a worker function
            def tokenize_text(text):
                tokens = word_tokenize(text.lower())
                return [token for token in tokens if token.strip()]
            
            # Use multiprocessing for faster tokenization on large datasets
            if len(X) > 10000:
                with Pool(processes=cpu_count()) as pool:
                    self.data = list(tqdm(pool.imap(tokenize_text, X), 
                                        total=len(X), 
                                        desc="Tokenizing text"))
            else:
                # Direct processing for smaller datasets to avoid overhead
                for i, text in enumerate(tqdm(X, desc="Tokenizing text")):
                    self.data[i] = tokenize_text(text)
        
        # Calculate max length efficiently
        lengths = [len(tokens) for tokens in self.data]
        self.max_length = max(lengths)
        
        # Print length distribution for better decision-making
        #percentiles = np.percentile(lengths, [50, 75, 90, 95, 99])
        #print(f"Sentence length stats: mean={np.mean(lengths):.1f}, median={percentiles[0]:.1f}")
        #print(f"75th={percentiles[1]:.1f}, 90th={percentiles[2]:.1f}, 95th={percentiles[3]:.1f}, 99th={percentiles[4]:.1f}, max={self.max_length}")
        
        # Suggest a reasonable max_length that covers most cases
        #suggested_max = int(percentiles[2])  # 90th percentile is often a good choice
        #print(f"Suggested max_length: {suggested_max} (covers 90% of samples)")
        
        # Optional: Allow setting a custom max_length to avoid outliers
        # self.max_length = suggested_max  # Uncomment to use suggested length
        
        # Print the first 10 tokenized examples efficiently
        #print("\nFirst 10 tokenized examples:")
        #for i in range(min(10, len(self.data))):
        #    print(f"Example {i+1} ({len(self.data[i])} tokens): {self.data[i][:10]}...")
            
    def __len__(self):
        """
        Must return the length of the dataset, so the dataloader can know
        how to split it into batches

        Returns:
            (int): the length of the dataset
        """

        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int): Index of the item to retrieve

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (int): the class label
                * length (int): the length (tokens) of the sentence
        """
        example = self.data[index]
        # map tokens to ids according to word2idx
        example = [self.word2idx.get(token, self.word2idx['<unk>']) for token in example]
        label = self.labels[index]
        length = len(example) 

        # zero padding using the maximum length from initialization
        # or truncation in case a larger example is found
        if length < self.max_length:
            example += [0] * (self.max_length - length)
        else:
            example = example[:self.max_length]
            
        #if index < 5:
        #    tokens = self.data[index]
        #    print(f"\nOriginal tokens: {tokens}")
        #    print(f"Encoded example: {example}")
        #    print(f"Label: {label}")
        #    print(f"Length: {min(length, self.max_length)}")
        
        return np.array(example), label, length