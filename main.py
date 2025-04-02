import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
import time
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from attention import SimpleSelfAttentionModel, MultiHeadAttentionModel, TransformerEncoderModel
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.metrics import accuracy_score, f1_score, recall_score
from early_stopper import EarlyStopper

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN, LSTM
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from torch.utils.tensorboard import SummaryWriter
from training import get_metrics_report, torch_train_val_split

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################

# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "crawl-300d-2M-subword.vec")
#EMBEDDINGS = os.path.join(EMB_PATH, "glove.twitter.27B.200d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 200

EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "MR"  # options: "MR", "Semeval2017A"

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configure seed for reproducibility
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
if torch.backends.cudnn.is_available():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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
train_loader, val_loader = torch_train_val_split(train_set, BATCH_SIZE, BATCH_SIZE)

#train_loader = DataLoader(
#    dataset=train_set,
#    batch_size=BATCH_SIZE,
#    shuffle=True,
#    num_workers=0,
#    pin_memory=torch.cuda.is_available()
#) # EX7

#test_loader = DataLoader(
#    dataset=test_set,
#    batch_size=BATCH_SIZE,
#    shuffle=False,
#    num_workers=0,
#    pin_memory=torch.cuda.is_available()
#) # EX7


#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################

#models = {
#    'BaselineDNN': BaselineDNN(
#        output_size=n_classes,
#        embeddings=embeddings,
#        trainable_emb=EMB_TRAINABLE
#    ),
#    'LSTM': LSTM(
#        output_size=n_classes,
#        embeddings=embeddings,
#        trainable_emb=EMB_TRAINABLE,
#        bidirectional=True,
#        dropout=0.2
#    ),
#    'SimpleSelfAttentionModel': SimpleSelfAttentionModel(
#        output_size=n_classes,
#        embeddings=embeddings,
#        max_length=train_set.max_length,
#        dropout=0.2,
#    ),
#    'MultiHeadAttentionModel': MultiHeadAttentionModel(
#        output_size=n_classes,
#        embeddings=embeddings,
#        max_length=train_set.max_length,
#        n_head=5,
#        dropout=0.0
#    ),
#    'TransformerEncoderModel': TransformerEncoderModel(
#        output_size=n_classes,
#        embeddings=embeddings,
#        max_length=train_set.max_length,
#        n_head=10,
#        n_layer=5,
#    )
#}
#
#
#model = models['MultiHeadAttentionModel']  # EX8

#model = BaselineDNN(output_size=n_classes,  # EX8
#                    embeddings=embeddings,
#                    trainable_emb=EMB_TRAINABLE)
#
#model = LSTM(output_size=n_classes,  # EX8
#                    embeddings=embeddings,
#                    trainable_emb=EMB_TRAINABLE,
#                    bidirectional=True,
#                    dropout=0.2)
#
#model = SimpleSelfAttentionModel(
#    embeddings=embeddings,
#    output_size=n_classes,
#    max_length=train_set.max_length,
#    dropout=0.2,
#)

model = MultiHeadAttentionModel(
    embeddings=embeddings,
    output_size=n_classes,
    max_length=train_set.max_length,
    n_head=5,
    dropout=0.0
)

#model = TransformerEncoderModel(
#    embeddings=embeddings,
#    output_size=n_classes,
#    max_length=train_set.max_length,
#    n_head=4,
#    n_layer=5,
#)

# Choose an appropriate loss criterion based on the number of classes
if n_classes == 2:
    criterion = nn.BCEWithLogitsLoss() # EX8
else:
    criterion = nn.CrossEntropyLoss() # EX8

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
parameters = [p for p in model.parameters() if p.requires_grad] # EX8

# Use Adam optimizer which adapts learning rates automatically
optimizer = torch.optim.Adam(parameters, lr=0.001, weight_decay=1e-5) # EX8

# Add a scheduler to adjust the learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',          # Reduce LR when monitored value stops decreasing
    factor=0.1,          # Multiply LR by this factor
    patience=3,         # Number of epochs with no improvement after which LR will be reduced
    verbose=True,        # Print message when LR is reduced
    min_lr=1e-6          # Lower bound on the learning rate
)

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(f"Model moved to {DEVICE}")

#############################################################################
# Training Pipeline
#############################################################################

# Setup early stopping
output_dir = f"outputs/{DATASET}"
os.makedirs(output_dir, exist_ok=True)
# Initialize TensorBoard writer

if model.__class__.__name__ == "LSTM":
    is_bidirectional = "BI_" if model.bidirectional else ""
    is_multilayered = "ML_" if model.num_layers > 1 else ""
else:
    is_bidirectional = ""
    is_multilayered = ""

log_dir = os.path.join("runs", f"{DATASET}_{is_multilayered}{is_bidirectional}{model.__class__.__name__}_{time.strftime('%Y%m%d-%H%M%S')}")
writer = SummaryWriter(log_dir)
print(f"TensorBoard logs will be saved to {log_dir}")

early_stopper = EarlyStopper(model, f'{output_dir}/{DATASET}_best_{is_multilayered}{is_bidirectional}{model.__class__.__name__}.pt', patience=5)

# Log model hyperparameters
writer.add_text("hyperparameters/model_type", f"{model.__class__.__name__}")
writer.add_text("hyperparameters/embeddings", EMBEDDINGS)
#writer.add_text("hyperparameters/dropout", str(model.dropout))
if model.__class__.__name__ == "LSTM":
    writer.add_text("hyperparameters/num_layers", str(model.num_layers))
if model.__class__.__name__ == "MultiHeadAttentionModel":
    writer.add_text("hyperparameters/n_head", str(model.n_head))
if model.__class__.__name__ == "TransformerEncoderModel":
    writer.add_text("hyperparameters/n_layer", str(model.n_layer))
writer.add_text("hyperparameters/dataset", DATASET)
writer.add_text("hyperparameters/batch_size", str(BATCH_SIZE))
writer.add_text("hyperparameters/embedding_dim", str(EMB_DIM))
writer.add_text("hyperparameters/epochs", str(EPOCHS))
writer.add_text("hyperparameters/learning_rate", str(optimizer.param_groups[0]['lr']))

# Setup early stopping

# Initialize tracking variables
best_test_loss = float('inf')
train_losses = []
val_losses = []

print(f"Starting training for {DATASET} dataset with {n_classes} classes")

for epoch in range(1, EPOCHS + 1):
    # Train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer)
    
    # Evaluate on train and test sets
    train_loss, (y_train_pred, y_train_gold) = eval_dataset(train_loader, model, criterion)
    train_losses.append(train_loss)
    val_loss, (y_test_pred, y_test_gold) = eval_dataset(val_loader, model, criterion)
    val_losses.append(val_loss)
    
    # Update the learning rate scheduler
    scheduler.step(val_loss)
    
    # Calculate metrics
    train_acc = accuracy_score(y_train_gold, y_train_pred)
    train_f1 = f1_score(y_train_gold, y_train_pred, average='macro')
    train_recall = recall_score(y_train_gold, y_train_pred, average='macro')
    
    test_acc = accuracy_score(y_test_gold, y_test_pred)
    test_f1 = f1_score(y_test_gold, y_test_pred, average='macro')
    test_recall = recall_score(y_test_gold, y_test_pred, average='macro')
    
    # Log metrics to TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/train', train_acc, epoch)
    writer.add_scalar('Accuracy/val', test_acc, epoch)
    writer.add_scalar('F1_Score/train', train_f1, epoch)
    writer.add_scalar('F1_Score/val', test_f1, epoch)
    writer.add_scalar('Recall/train', train_recall, epoch)
    writer.add_scalar('Recall/val', test_recall, epoch)
    
    # Print metrics for current epoch
    print()
    print("Training metrics:")
    print(get_metrics_report(y_train_gold, y_train_pred))
    print("Validation metrics:")
    print(get_metrics_report(y_test_gold, y_test_pred))
    print()
    
    # Print metrics for current epoch
    #print(f"\nEpoch {epoch}/{EPOCHS}")
    #print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
    #print(f"Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}")
    
    # Check for early stopping
    if early_stopper.early_stop(val_loss):
        print(f"Early stopping triggered at epoch {epoch}")
        print(f'Epoch {epoch}/{EPOCHS}, Loss at training set: {train_loss}\n\tLoss at validation set: {val_loss}')
        print('Training has been completed.\n')
        break

# Load the best model for final evaluation
model.load_state_dict(torch.load(f'{output_dir}/{DATASET}_best_{is_multilayered}{is_bidirectional}{model.__class__.__name__}.pt'))
print("\nLoaded best model for final evaluation")

# Final evaluation on test set
_, (y_test_pred, y_test_gold) = eval_dataset(val_loader, model, criterion)

# Calculate final metrics
final_acc = accuracy_score(y_test_gold, y_test_pred)
final_f1 = f1_score(y_test_gold, y_test_pred, average='macro')
final_recall = recall_score(y_test_gold, y_test_pred, average='macro')

print(f"\nFinal evaluation on {DATASET} dataset:")
print(f"  Accuracy: {final_acc:.4f}")
print(f"  F1 Score (macro): {final_f1:.4f}")
print(f"  Recall (macro): {final_recall:.4f}")

# Create and save confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_gold, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title(f'Confusion Matrix - {DATASET}')
plt.tight_layout()
plt.savefig(f'{output_dir}/confusion_matrix.png')
print(f"Confusion matrix saved to {output_dir}/confusion_matrix.png")

# Plot loss curves
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title(f'Learning Curves - {DATASET}')
plt.legend()
plt.grid(True)
plt.savefig(f'{output_dir}/learning_curves.png')
print(f"Learning curves saved to {output_dir}/learning_curves.png")

# Close TensorBoard writer
writer.close()

print(f"\nTraining completed. Results saved to {output_dir}")
print("To view TensorBoard logs, run: tensorboard --logdir=runs")