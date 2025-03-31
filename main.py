import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.utils.data import DataLoader
import wandb

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, recall_score
from early_stopper import EarlyStopper

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from training import get_metrics_report

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "MR"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
le = LabelEncoder()
y_train = le.fit_transform(y_train) # EX1
y_test =  le.transform(y_test) # EX1
n_classes = le.classes_.size # EX1 - LabelEncoder.classes_.size

print("First 10 original labels:", y_train[:10])
print("Label mappings:", {i: label for i, label in enumerate(le.classes_)})

# Define our PyTorch-based Dataset
train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

# EX7 - Define our PyTorch-based DataLoader
train_loader = DataLoader(
    dataset=train_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
) # EX7

test_loader = DataLoader(
    dataset=test_set,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=torch.cuda.is_available()
) # EX7


#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
model = BaselineDNN(output_size=n_classes,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)

# Choose an appropriate loss criterion based on the number of classes
if n_classes == 2:
    criterion = nn.BCEWithLogitsLoss() # EX8
else:
    criterion = nn.CrossEntropyLoss() # EX8

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
parameters = [p for p in model.parameters() if p.requires_grad] # EX8

# Use Adam optimizer which adapts learning rates automatically
optimizer = torch.optim.Adam(parameters, lr=0.001, weight_decay=1e-5) # EX8

# Add these before the training loop
train_losses = []
test_losses = []
train_metrics = []
test_metrics = []


# move the mode weight to cpu or gpu
model.to(DEVICE)
print(f"Model moved to {DEVICE}")

#############################################################################
# Training Pipeline
#############################################################################
wandb.init(
    project="sentiment-analysis",
    name=f"{DATASET}-baseline-dnn",
    config={
        "architecture": "BaselineDNN",
        "dataset": DATASET,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "embedding_dim": EMB_DIM,
        "trainable_emb": EMB_TRAINABLE,
        "learning_rate": optimizer.param_groups[0]['lr'],
        "classes": list(le.classes_)
    }
)

# Log model architecture
wandb.watch(model, log="all")

# Setup early stopping
early_stopper = EarlyStopper(model, f'{DATASET}_best_model.pt', patience=5)

print(f"Starting training for {DATASET} dataset with {n_classes} classes")

for epoch in range(1, EPOCHS + 1):
    # Train the model for one epoch
    train_loss = train_dataset(epoch, train_loader, model, criterion, optimizer)
    
    # Evaluate on train and test sets
    train_loss, (y_train_pred, y_train_gold) = eval_dataset(train_loader, model, criterion)
    test_loss, (y_test_pred, y_test_gold) = eval_dataset(test_loader, model, criterion)
    
    # Calculate metrics
    train_metrics = {
        "train/loss": train_loss,
        "train/accuracy": accuracy_score(y_train_gold, y_train_pred),
        "train/f1_macro": f1_score(y_train_gold, y_train_pred, average='macro'),
        "train/recall_macro": recall_score(y_train_gold, y_train_pred, average='macro')
    }
    
    test_metrics = {
        "test/loss": test_loss,
        "test/accuracy": accuracy_score(y_test_gold, y_test_pred),
        "test/f1_macro": f1_score(y_test_gold, y_test_pred, average='macro'),
        "test/recall_macro": recall_score(y_test_gold, y_test_pred, average='macro')
    }
    
    # Log metrics to wandb
    wandb.log({**train_metrics, **test_metrics, "epoch": epoch})
    
    # Print metrics
    print(f"\nEpoch {epoch}/{EPOCHS}")
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['train/accuracy']:.4f}, F1: {train_metrics['train/f1_macro']:.4f}")
    print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_metrics['test/accuracy']:.4f}, F1: {test_metrics['test/f1_macro']:.4f}")
    
    # Check for early stopping
    if early_stopper.early_stop(test_loss):
        print(f"Early stopping triggered at epoch {epoch}")
        break

# Load the best model for final evaluation
model.load_state_dict(torch.load(f'{DATASET}_best_model.pt'))
print("\nLoaded best model for final evaluation")

# Final evaluation on test set
_, (y_test_pred, y_test_gold) = eval_dataset(test_loader, model, criterion)

# Log final confusion matrix to wandb
cm = confusion_matrix(y_test_gold, y_test_pred)
wandb.log({
    "confusion_matrix": wandb.plot.confusion_matrix(
        probs=None,
        y_true=y_test_gold,
        preds=y_test_pred,
        class_names=le.classes_
    )
})

# Log final metrics
final_metrics = {
    "final/accuracy": accuracy_score(y_test_gold, y_test_pred),
    "final/f1_macro": f1_score(y_test_gold, y_test_pred, average='macro'),
    "final/recall_macro": recall_score(y_test_gold, y_test_pred, average='macro')
}
wandb.log(final_metrics)

# Print final report
print(f"\nFinal evaluation on {DATASET} dataset:")
print(get_metrics_report([y_test_gold], [y_test_pred]))

# Save model to wandb
torch.save(model.state_dict(), f"{DATASET}_final_model.pt")
wandb.save(f"{DATASET}_final_model.pt")

# Close wandb run
wandb.finish()

print("Training completed. Results logged to Weights & Biases project 'sentiment-analysis'")